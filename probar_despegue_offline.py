"""Alineación del yaw del EKF y registro antes/durante el despegue; sin dron."""
import json
import os
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import numpy as np
from banco_headless import Banco, extraer

MONTAJE = 5.2


def main():
    b = Banco()
    # Posición del EKF: objetos falsos y tiempos cortos, sin importar hardware.
    Pekf = NS(mocap_pose={'x': 1.0, 'y': 2.0, 'z': 0.1},
              stop_event=threading.Event())
    listo = extraer('Prueba_ruta_carrot.py', ['_esperar_ekf_listo'],
                    dict(P=Pekf, time=time))['_esperar_ekf_listo']

    def caso_ekf(nombre, pose, esperado, minimo, cambios=(), maximo=1.5):
        Pekf.stop_event.clear()
        scf = NS(pose_data=pose.copy())
        marcas = {}

        def cambiar(etiqueta, nueva_pose):
            scf.pose_data = nueva_pose.copy()
            marcas[etiqueta] = time.perf_counter()

        timers = [threading.Timer(t, cambiar, args=(etiqueta, nueva_pose))
                  for t, etiqueta, nueva_pose in cambios]
        inicio = time.perf_counter()
        for timer in timers:
            timer.start()
        try:
            resultado = listo(scf, 0.2, 1.0, 0.03, 0.2)
            fin = time.perf_counter()
        finally:
            for timer in timers:
                timer.cancel()
                timer.join()
        elapsed = fin - inicio
        b.comprobar(nombre, resultado is esperado and minimo <= elapsed < maximo,
                    (esperado, f'{minimo} <= tiempo < {maximo} s'),
                    (resultado, elapsed))
        return fin, marcas

    igual = Pekf.mocap_pose.copy()
    caso_ekf('EKF = MoCap: espera mínima y estabilidad posterior',
             igual, True, 0.4)
    for eje in ('x', 'y', 'z'):
        desviado = dict(igual, **{eje: igual[eje] + 0.10})
        caso_ekf(f'EKF desviado 10 cm en {eje}: agota el plazo',
                 desviado, False, 1.0)
    desviado = dict(igual, x=igual['x'] + 0.10)
    fin, marcas = caso_ekf('EKF converge a mitad de la espera',
                           desviado, True, 0.7, [(0.5, 'converge', igual)])
    b.comprobar('convergencia exige 0.2 s estables',
                'converge' in marcas and fin - marcas['converge'] >= 0.19,
                '>=0.2 s (margen de medición 0.01 s)',
                fin - marcas.get('converge', fin))
    fin, marcas = caso_ekf('EKF sale de tolerancia: reinicia estabilidad',
                           desviado, True, 0.85,
                           [(0.25, 'entra', igual), (0.35, 'sale', desviado),
                            (0.65, 'vuelve', igual)])
    b.comprobar('intervalo interrumpido no cuenta para el segundo',
                'vuelve' in marcas and fin - marcas['vuelve'] >= 0.19,
                '>=0.2 s desde la segunda entrada (margen 0.01 s)',
                fin - marcas.get('vuelve', fin))
    for demora in (0.1, 0.3):
        Pekf.stop_event.clear()
        timer = threading.Timer(demora, Pekf.stop_event.set)
        inicio = time.perf_counter()
        timer.start()
        try:
            resultado = listo(NS(pose_data=igual.copy()), 0.2, 1.0, 0.03, 0.2)
            elapsed = time.perf_counter() - inicio
        finally:
            timer.cancel()
            timer.join()
        b.comprobar(f'paro durante espera EKF a los {demora} s',
                    resultado is False and elapsed < demora + 0.3,
                    'False sin esperar el timeout', (resultado, elapsed))
    caso_ekf('EKF sin pose_data: no despega', {}, False, 1.0)

    # Rampa pura: extremos, punto medio y monotonía (incluye saturación).
    rampa = extraer('Prueba_ruta_carrot.py', ['_rampa_inicio'])['_rampa_inicio']
    for nombre, t, esperado in [('antes del inicio', 9.0, 0.0),
                                ('en t0', 10.0, 0.0),
                                ('a mitad', 11.5, 0.5),
                                ('al completar', 13.0, 1.0),
                                ('después del final', 14.0, 1.0)]:
        valor = rampa(t, 10.0, 3.0)
        b.comprobar('rampa ' + nombre, valor == esperado, esperado, valor)
    valores = [rampa(9.0 + i * 0.05, 10.0, 3.0) for i in range(101)]
    b.comprobar('rampa monótona y acotada',
                all(0.0 <= v <= 1.0 for v in valores)
                and all(a <= c for a, c in zip(valores, valores[1:])),
                'creciente en [0, 1]', (min(valores), max(valores)))
    for duracion in (0.0, -1.0):
        valor = rampa(9.0, 10.0, duracion)
        b.comprobar(f'rampa duración {duracion}: inmediata',
                    valor == 1.0, 1.0, valor)

    # Visión falsa, sin cargar modelos ni importar módulos de hardware.
    Pvision = NS(stop_event=threading.Event())
    esperar = extraer('Prueba_ruta_carrot.py',
                      ['_formato_camara', '_esperar_vision_lista'],
                      dict(P=Pvision, time=time, EVASION_MODO='esquivar',
                           VISION_MODELO='midas', EVASION_CAL_FORMATO='raw')
                      )['_esperar_vision_lista']
    muestras = []

    def ultimo_nuevo():
        muestras.append(len(muestras) + 1)
        return {'mapa_ts': muestras[-1]}

    receptor = NS(formatos={0: 10, 1: 0})
    vis = NS(midas_listo=True, ultimo=ultimo_nuevo, receptor=receptor)
    inicio = time.perf_counter()
    resultado = esperar(vis, 1.0)
    elapsed = time.perf_counter() - inicio
    b.comprobar('visión lista con dos cambios de mapa: True rápido',
                resultado is True and len(muestras) >= 3 and elapsed < 0.5,
                'True, >=2 cambios, <0.5 s', (resultado, len(muestras), elapsed))

    vis.midas_listo = False
    inicio = time.perf_counter()
    resultado = esperar(vis, 0.15)
    elapsed = time.perf_counter() - inicio
    b.comprobar('modelo sin cargar: agota la espera sin despegar',
                resultado is False and 0.15 <= elapsed < 0.75,
                'False, timeout 0.15 s', (resultado, elapsed))

    vis = NS(midas_listo=True, ultimo=lambda: {'mapa_ts': 42.0}, receptor=receptor)
    resultado = esperar(vis, 0.15)
    b.comprobar('modelo cargado con mapa congelado: no despega',
                resultado is False, False, resultado)
    Pvision.stop_event.set()
    vis.ultimo = ultimo_nuevo
    inicio = time.perf_counter()
    resultado = esperar(vis, 1.0)
    elapsed = time.perf_counter() - inicio
    b.comprobar('paro con visión lista: no despega ni espera el timeout',
                resultado is False and elapsed < 0.5,
                'False inmediato', (resultado, elapsed))

    stop = threading.Event()
    P = NS(mocap_pose={'yaw': 0.0}, _mocap_ori={'w': 1}, stop_event=stop)
    tmp = tempfile.mkdtemp()
    scope = extraer('Prueba_ruta_carrot.py',
                    ['_wrap', '_yaw_ekf', '_yaw_mundo_del_mocap',
                     '_esperar_yaw_alineado', '_alinear_yaw_ekf',
                     '_guardar_registro_despegue'],
                    dict(P=P, time=time, YAW_MONTAJE_DEG=MONTAJE,
                         ALINEA_YAW_CONV_S=0.0, __file__=os.path.join(tmp, 'x.py')))
    guardia, alinear = scope['_esperar_yaw_alineado'], scope['_alinear_yaw_ekf']

    def real(deg, ori=True):
        """Pone en el MoCap un dron cuyo rumbo real es `deg` (el MoCap ve +montaje)."""
        P.mocap_pose['yaw'] = None if deg is None else np.radians(deg + MONTAJE)
        P._mocap_ori = {'w': 1} if ori else None

    def caso(nombre, esperado, ekf_deg, real_deg, ori=True, espera=0.4):
        real(real_deg, ori)
        scf = NS(pose_data={} if ekf_deg is None else {'yaw': ekf_deg})
        r = guardia(scf, 10.0, espera)
        b.comprobar(nombre, r is esperado, esperado, r)

    # ── Chequeo previo al despegue (contra el rumbo REAL, sin el montaje) ──
    caso('alineado (Δ2°)', True, 30.0, 32.0)
    caso('dron puesto torcido 40°', False, 0.0, 40.0)
    caso('dron torcido 180°', False, 0.0, 180.0)
    caso('cruce ±180° (Δ2°)', True, 179.0, -179.0)
    caso('sin orientación del MoCap: no se puede medir', True, 0.0, 40.0, ori=False)
    caso('sin yaw del EKF: no se despega', False, None, 10.0)

    real(40.0)
    scf = NS(pose_data={'yaw': 0.0})
    threading.Timer(0.3, lambda: scf.pose_data.update(yaw=39.0)).start()
    r = guardia(scf, 10.0, 2.0)
    b.comprobar('converge durante la espera', r is True, True, r)

    # ── Reinicio del EKF con el rumbo real ──
    def dron(con_param=True, falla=False):
        llamadas = []
        def set_value(nombre, valor):
            if falla:
                raise RuntimeError('radio caída')
            llamadas.append((nombre, valor))
        toc = {'kalman': {'initialYaw': object()}} if con_param else {'kalman': {}}
        return NS(cf=NS(param=NS(toc=NS(toc=toc), set_value=set_value))), llamadas

    real(40.0)
    scf, ll = dron()
    r = alinear(scf)
    yaw_fijado = float(ll[0][1]) if ll else None
    b.comprobar('reinicia el EKF con initialYaw = rumbo real (rad)',
                r is True and [n for n, _ in ll] == [
                    'kalman.initialYaw', 'kalman.resetEstimation',
                    'kalman.resetEstimation'] and [v for _, v in ll[1:]] == ['1', '0']
                and abs(yaw_fijado - np.radians(40.0)) < 1e-5,
                f'{np.radians(40.0):.5f} rad', yaw_fijado)
    scf, ll = dron(con_param=False)
    b.comprobar('firmware sin initialYaw: no toca nada', alinear(scf) is False
                and ll == [], False, ll)
    real(40.0, ori=False)
    scf, ll = dron()
    b.comprobar('sin orientación del MoCap: no reinicia', alinear(scf) is False
                and ll == [], False, ll)
    real(40.0)
    scf, ll = dron(falla=True)
    b.comprobar('falla el radio: devuelve False sin romper', alinear(scf) is False,
                False, 'excepción contenida')

    # ── Registro del ascenso ──
    scope['_guardar_registro_despegue']([(0.0, 1.0, 2.0, 0.0, 0.1, -0.2, 3.0),
                                         (0.02, 1.0, 2.0, 0.01, None, None, None)])
    ruta = os.path.join(tmp, 'capturas_vuelo')
    archivos = os.listdir(ruta) if os.path.isdir(ruta) else []
    datos = json.load(open(os.path.join(ruta, archivos[0]))) if archivos else {}
    b.comprobar('guarda el registro del despegue',
                len(archivos) == 1 and len(datos.get('columnas', [])) == 7
                and len(datos.get('filas', [])) == 2, '1 archivo, 7 col, 2 filas',
                (len(archivos), len(datos.get('filas', []))))

    # ── Motive tiene que ver el dron antes de despegar ──
    Pv = NS(_t_mocap_cambio=[0.0], MOCAP_CONGELADO_S=0.25,
            stop_event=threading.Event())
    vivo = extraer('Prueba_ruta_carrot.py', ['_esperar_mocap_vivo'],
                   dict(P=Pv, time=time))['_esperar_mocap_vivo']
    fin = threading.Event()

    def ruido():                       # Motive rastreando: la pose cambia siempre
        while not fin.is_set():
            Pv._t_mocap_cambio[0] = time.perf_counter()
            time.sleep(0.011)
    h = threading.Thread(target=ruido); h.start()
    r = vivo(2.0)
    fin.set(); h.join()
    b.comprobar('MoCap rastreando (pose cambia): se despega', r is True, True, r)

    Pv._t_mocap_cambio[0] = time.perf_counter() - 5.0    # congelado hace 5 s
    t0 = time.time(); r = vivo(0.6)
    b.comprobar('MoCap congelado: no se despega y no se queda esperando',
                r is False and time.time() - t0 < 1.0, False, r)

    fin.clear()
    Pv._t_mocap_cambio[0] = time.perf_counter() - 5.0
    threading.Timer(0.3, lambda: threading.Thread(target=ruido).start()).start()
    r = vivo(3.0)
    fin.set()
    b.comprobar('Motive vuelve a verlo durante la espera: se despega', r is True, True, r)

    # ── Paro de emergencia durante la espera: no despega ──
    stop.set()
    real(40.0)
    r = guardia(NS(pose_data={'yaw': 0.0}), 10.0, 2.0)
    b.comprobar('paro de emergencia: no despega', r is False, False, r)
    return b.terminar()


if __name__ == '__main__':
    raise SystemExit(main())
