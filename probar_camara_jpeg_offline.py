"""JPEG Bayer y compuerta de calibración; sin hardware ni modelos."""
import ast
import io
import queue
import threading
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import numpy as np

from banco_headless import Banco, extraer


def constantes(archivo, nombres, entorno=None):
    scope = dict(entorno or {})
    tree = ast.parse(Path(archivo).read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in nombres
                        for t in node.targets)):
            if entorno is None:
                scope[node.targets[0].id] = ast.literal_eval(node.value)
            else:
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             archivo, 'exec'), scope)
    return scope


def main():
    b = Banco()
    archivo = 'sistema_vuelo.py'
    env = constantes(archivo, ['PATRONES_BAYER', 'DEMOSAICO',
                               'VISION_JPEG_BAYER_GRIS', 'VISION_AJUSTES'],
                     dict(cv2=cv2, np=np, threading=threading, _HAY_CV2=True))
    scope = extraer(archivo, ['HiloReceptorAIDeck'], env)
    receptor = scope['HiloReceptorAIDeck'](
        'sin-conexion', 0, queue.Queue(), bayer=True,
        patron=env['VISION_AJUSTES']['patron'])
    b.comprobar('receptor construido sin conectar ni arrancar hilo',
                receptor.sock is None and not receptor.is_alive(),
                'sin socket/hilo activo', (receptor.sock, receptor.is_alive()))

    alto, ancho = 96, 128
    color = np.zeros((alto, ancho, 3), dtype=np.uint8)
    color[:] = (0, 0, 240)
    color[:, 64:] = (30, 220, 40)
    color[64:, 64:] = (220, 40, 20)
    # BG por defecto: mosaico físico B G / G R.
    canales = {'BG': ((0, 1), (1, 2)), 'GB': ((1, 0), (2, 1)),
               'RG': ((2, 1), (1, 0)), 'GR': ((1, 2), (0, 1))}
    mosaico = np.empty((alto, ancho), dtype=np.uint8)
    for y in range(2):
        for x in range(2):
            mosaico[y::2, x::2] = color[y::2, x::2, canales[receptor.patron][y][x]]
    ok, jpg = cv2.imencode('.jpg', mosaico, [cv2.IMWRITE_JPEG_QUALITY, 100])
    if not ok:
        raise RuntimeError('No se pudo codificar el JPEG sintético')
    buffer = jpg.tobytes()
    limpio = receptor._decodificar(buffer, 1, ancho, alto)
    b.comprobar('JPEG Bayer: gris uint8 de tres canales y tamaño correcto',
                limpio.shape == (alto, ancho, 3) and limpio.dtype == np.uint8
                and np.array_equal(limpio[:, :, 0], limpio[:, :, 1])
                and np.array_equal(limpio[:, :, 1], limpio[:, :, 2]),
                '(96,128,3), uint8, canales iguales', (limpio.shape, limpio.dtype))
    scope['VISION_JPEG_BAYER_GRIS'] = False
    anterior = receptor._decodificar(buffer, 1, ancho, alto)

    def vecinos(img):
        zona = img[12:52, 12:52, 0].astype(float)
        return (np.abs(np.diff(zona, axis=0)).mean()
                + np.abs(np.diff(zona, axis=1)).mean()) / 2.0

    antes, despues = vecinos(anterior), vecinos(limpio)
    b.comprobar('ajedrez en rojo plano: diferencia vecinal cae al menos 90%',
                antes > 20.0 and despues < antes * 0.1, antes, despues)
    gris_jpg = cv2.imdecode(jpg, cv2.IMREAD_UNCHANGED)
    esperado = cv2.cvtColor(gris_jpg, cv2.COLOR_GRAY2BGR)
    b.comprobar('opción False conserva JPEG monocanal anterior',
                np.array_equal(anterior, esperado), 'GRAY2BGR directo', 'comparación exacta')
    scope['VISION_JPEG_BAYER_GRIS'] = True
    receptor.bayer = False
    mono = receptor._decodificar(buffer, 1, ancho, alto)
    b.comprobar('cámara mono conserva JPEG sin demosaico',
                np.array_equal(mono, esperado), 'GRAY2BGR directo', 'comparación exacta')
    receptor.bayer = True
    for opcion in (False, True):
        scope['VISION_JPEG_BAYER_GRIS'] = opcion
        raw = receptor._decodificar(mosaico.tobytes(), 0, ancho, alto)
        esperado_raw = cv2.cvtColor(mosaico, env['DEMOSAICO'][0][1][receptor.patron])
        b.comprobar(f'RAW intacto con opción {opcion}',
                    np.array_equal(raw, esperado_raw), 'demosaico RAW original', 'igual píxel a píxel')
        ok, jpg_color = cv2.imencode('.jpg', color)
        if not ok:
            raise RuntimeError('No se pudo codificar JPEG color')
        salida = receptor._decodificar(jpg_color.tobytes(), 1, ancho, alto)
        esperado_color = cv2.imdecode(jpg_color, cv2.IMREAD_UNCHANGED)
        b.comprobar(f'JPEG de tres canales intacto con opción {opcion}',
                    np.array_equal(salida, esperado_color), 'imdecode sin cambios', 'igual píxel a píxel')

    # Reloj simulado: verifica espera sin sleeps ni conexiones.
    class Reloj:
        def __init__(self):
            self.t = 0.0

        def perf_counter(self):
            return self.t

        def is_set(self):
            return False

        def wait(self, dt):
            self.t += dt

    config = constantes('volar_ruta.py',
                        ['EVASION_MODO', 'VISION_MODELO', 'EVASION_CAL_FORMATO'])
    reloj = Reloj()
    config.update(P=NS(stop_event=reloj), time=reloj)
    guardia = extraer('volar_ruta.py',
                      ['_formato_camara', '_esperar_vision_lista'], config)
    for nombre, cuentas, cal, modelo, modo, esperado in [
            ('RAW coincide', {0: 9, 1: 2}, 'raw', 'midas', 'esquivar', True),
            ('JPEG coincide', {0: 2, 1: 9}, 'jpeg', 'hibrido', 'esquivar', True),
            ('JPEG no coincide', {0: 2, 1: 9}, 'raw', 'hibrido', 'esquivar', False),
            ('RAW no coincide en observar', {0: 9, 1: 2}, 'jpeg', 'midas', 'observar', False),
            ('sin frames espera', {0: 0, 1: 0}, 'raw', 'midas', 'esquivar', False),
            ('métrico exento', {0: 0, 1: 9}, 'raw', 'depth_anything_v2_metrico', 'esquivar', True),
            ('evasión desactivada', {0: 0, 1: 9}, 'raw', 'midas', 'off', True)]:
        reloj.t = 0.0
        receptor.formatos = cuentas
        contador = iter(range(100))
        vis = NS(receptor=receptor, midas_listo=True,
                 ultimo=lambda: {'mapa_ts': next(contador)})
        guardia.update(EVASION_CAL_FORMATO=cal, VISION_MODELO=modelo, EVASION_MODO=modo)
        salida = io.StringIO()
        with redirect_stdout(salida):
            resultado = guardia['_esperar_vision_lista'](vis, 0.2)
        mensaje = salida.getvalue()
        b.comprobar(nombre, resultado is esperado, esperado, resultado)
        if 'no coincide' in nombre:
            formato = guardia['_formato_camara'](vis)
            b.comprobar(nombre + ': mensaje accionable',
                        f'la calibración es de {cal}' in mensaje
                        and f'la cámara manda {formato}' in mensaje
                        and '--calibrar-midas' in mensaje
                        and f'EVASION_CAL_FORMATO = {formato!r}' in mensaje,
                        'origen, destino y corrección', mensaje.strip())
        if nombre == 'sin frames espera':
            b.comprobar('sin frames no decide por incompatibilidad: agota espera',
                        reloj.t >= 0.2 and 'recalibra' not in mensaje,
                        'timeout sin diagnóstico de formato', (reloj.t, mensaje.strip()))
        etiqueta = receptor.formato_dominante()
        detectado = guardia['_formato_camara'](vis)
        b.comprobar(nombre + ': etiqueta del monitor',
                    etiqueta == (detectado.upper() if detectado else '?'),
                    detectado, etiqueta)
    return b.terminar()


if __name__ == '__main__':
    raise SystemExit(main())
