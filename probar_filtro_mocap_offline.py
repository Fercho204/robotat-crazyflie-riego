"""Filtro de saltos del MoCap (_filtrar_salto_mocap); sin dron ni MoCap real."""
import threading
import time
from types import SimpleNamespace
from banco_headless import Banco, extraer


def main():
    b = Banco()
    paros = []
    stop_event = threading.Event()

    def parar(origen):
        paros.append(origen)
        stop_event.set()          # el efecto real que importa aquí

    def entorno():
        return dict(time=time, stop_event=stop_event,
                   activar_paro_emergencia=parar,
                   MOCAP_SALTO_MAX_VEL=3.0, MOCAP_SALTO_DT_MAX=0.1,
                   MOCAP_SALTO_DT_MIN=0.04,
                   MOCAP_SALTO_DURACION_PARO_S=0.5,
                   _mocap_gate={'x': None, 'y': None, 'z': None, 't': None,
                               't_primer_rechazo': None})

    scope = extraer('sistema_vuelo.py', ['_filtrar_salto_mocap'], entorno())
    f = scope['_filtrar_salto_mocap']

    # Ráfaga real de llegadas: 2 cm inmediatamente, sin sleep.
    scope['_mocap_gate'].update(x=0.0, y=0.0, z=0.5, t=time.perf_counter(),
                                t_primer_rechazo=None)
    muestra = f(0.02, 0.0, 0.5)
    b.comprobar('rafaga MQTT: 2 cm sin espera se aceptan',
                muestra == (0.02, 0.0, 0.5), (0.02, 0.0, 0.5), muestra)

    # Tiempo simulado para no depender de la planificación del sistema.
    scope['time'] = SimpleNamespace(perf_counter=lambda: 10.021)
    scope['_mocap_gate'].update(x=0.0, y=0.0, z=0.5, t=10.0,
                                t_primer_rechazo=None)
    muestra = f(0.187, 0.0, 0.5)
    b.comprobar('salto de 18.7 cm en 21 ms sigue rechazado',
                muestra == (0.0, 0.0, 0.5), (0.0, 0.0, 0.5), muestra)
    scope['time'] = time
    scope['_mocap_gate'].update(x=None, y=None, z=None, t=None,
                                t_primer_rechazo=None)

    x, y, z = f(0.0, 0.0, 0.5)
    b.comprobar('primera muestra: se acepta tal cual', (x, y, z) == (0.0, 0.0, 0.5),
                (0.0, 0.0, 0.5), (x, y, z))

    time.sleep(0.02)
    x, y, z = f(0.187, 0.0, 0.567)     # el primer incidente: 18.7/6.7 cm en ~21 ms
    b.comprobar('salto de 8-9 m/s se descarta (repite la anterior)',
                (x, y, z) == (0.0, 0.0, 0.5), (0.0, 0.0, 0.5), (x, y, z))

    time.sleep(0.02)
    x, y, z = f(0.006, 0.0, 0.51)      # movimiento normal de vuelo
    b.comprobar('movimiento normal (≈30 cm/s) se acepta',
                (x, y, z) == (0.006, 0.0, 0.51), (0.006, 0.0, 0.51), (x, y, z))

    # El segundo incidente: el MoCap se congela y al volver llega un salto
    # grande. Ya no debe haber "tras N rechazos se acepta": para este dron,
    # ninguna cantidad de rechazos vuelve real una velocidad imposible.
    scope['_mocap_gate'].update(x=0.0, y=0.0, z=0.012, t=time.perf_counter(),
                                t_primer_rechazo=None)
    valores = set()
    for _ in range(30):
        time.sleep(0.02)                   # 30 x 20 ms = 0.6 s > el umbral
        valores.add(f(0.72, 0.0, 0.303))   # el salto real: 72 cm, repetido
    b.comprobar('nunca se acepta el salto por repetirse (sin válvula de escape)',
                valores == {(0.0, 0.0, 0.012)}, {(0.0, 0.0, 0.012)}, valores)

    # Rechazando en el sitio más de MOCAP_SALTO_DURACION_PARO_S: paro.
    b.comprobar('tras rechazar > 0.5 s seguido: paro de emergencia',
                len(paros) == 1 and 'MoCap' in paros[0], 1, len(paros))

    # Tras el paro, sigue descartando (no se queda mandando el paro sin fin).
    time.sleep(0.02)
    x, y, z = f(0.72, 0.0, 0.303)
    b.comprobar('tras el paro no se vuelve a avisar (stop_event ya puesto)',
                (x, y, z) == (0.0, 0.0, 0.012) and len(paros) == 1,
                (0.0, 0.0, 0.012), ((x, y, z), len(paros)))

    # El presupuesto (MOCAP_SALTO_MAX_VEL × tiempo) está topado en
    # MOCAP_SALTO_DT_MAX: tras un hueco largo sin muestras, lo que cabe en
    # ese tope (aquí 3.0 m/s × 0.1 s = 30 cm) se acepta igual...
    stop_event.clear()
    scope['_mocap_gate'].update(x=0.0, y=0.0, z=0.5, t=time.perf_counter() - 1.0,
                                t_primer_rechazo=None)
    x, y, z = f(0.2, 0.0, 0.5)          # 20 cm, cabe en el presupuesto topado
    b.comprobar('tras un hueco largo, un cambio pequeño (dentro del tope) se acepta',
                (x, y, z) == (0.2, 0.0, 0.5), (0.2, 0.0, 0.5), (x, y, z))

    # ...pero uno grande NO se vuelve plausible solo por haber esperado: es
    # justo el fallo del incidente 2 (esperar "suficiente" no debe bastar
    # para creerse un salto imposible).
    scope['_mocap_gate'].update(x=0.0, y=0.0, z=0.5, t=time.perf_counter() - 1.0,
                                t_primer_rechazo=None)
    x, y, z = f(0.5, 0.0, 0.5)          # 50 cm, no cabe aunque haya pasado 1 s
    b.comprobar('tras un hueco largo, uno grande sigue sin ser plausible',
                (x, y, z) == (0.0, 0.0, 0.5), (0.0, 0.0, 0.5), (x, y, z))

    # ── MoCap congelado: la pose se repite idéntica aunque lleguen mensajes ──
    cong = extraer('sistema_vuelo.py', ['_registrar_cambio_mocap'],
                   dict(MOCAP_CONGELADO_S=0.25, _t_mocap_cambio=[0.0],
                        _mocap_ult_crudo=[None]))
    reg = cong['_registrar_cambio_mocap']
    r1 = reg(0.1, 0.2, 0.117, 10.00)              # primera muestra
    r2 = reg(0.1, 0.2, 0.117, 10.02)              # idéntica: no es cambio
    b.comprobar('pose idéntica no cuenta como cambio',
                r1 is False and r2 is False and cong['_t_mocap_cambio'][0] == 10.00,
                10.00, cong['_t_mocap_cambio'][0])
    r3 = reg(0.1000071, 0.2, 0.117, 10.04)       # ruido real de Motive (7 µm)
    b.comprobar('ruido mínimo real sí cuenta como cambio (dron rastreado)',
                r3 is False and cong['_t_mocap_cambio'][0] == 10.04, 10.04,
                cong['_t_mocap_cambio'][0])
    r4 = reg(0.42, 0.2, 0.03, 13.60)              # tras 3.6 s idéntica, cambia
    b.comprobar('cambio tras > 0.25 s congelada = reenganche', r4 is True, True, r4)

    # En reenganche, la referencia vieja es la mala: se acepta la nueva en vez
    # de rechazarla para siempre (el vuelo de las 20:00 acabó en paro por eso).
    stop_event.clear()
    n_paros = len(paros)
    scope['_mocap_gate'].update(x=0.1, y=0.2, z=0.117, t=time.perf_counter(),
                                t_primer_rechazo=None)
    x, y, z = f(0.42, 0.2, 0.03, reenganche=True)
    b.comprobar('reenganche: se toma la posición nueva aunque salte',
                (x, y, z) == (0.42, 0.2, 0.03), (0.42, 0.2, 0.03), (x, y, z))
    time.sleep(0.02)
    x, y, z = f(0.421, 0.2, 0.031)
    b.comprobar('tras reenganchar, sigue normal y sin paro',
                (x, y, z) == (0.421, 0.2, 0.031) and len(paros) == n_paros,
                (0.421, 0.2, 0.031), ((x, y, z), len(paros) - n_paros))
    return b.terminar()


if __name__ == '__main__':
    raise SystemExit(main())
