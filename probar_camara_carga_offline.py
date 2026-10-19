"""Receptor de la cÃ¡mara bajo la carga de hilos del vuelo; sin hardware.

Un falso AI-deck (proceso aparte) manda frames RAW 324x244 en paquetes CPX a
~240 kB/s, como el enlace real, con la hora de envÃ­o escrita en la imagen.
Hilos Python ocupados imitan la radio, el MoCap y el control. Se mide quÃ©
llega a la cola de la visiÃ³n y con cuÃ¡nto retraso, con el receptor en este
proceso (como antes) y en puente_camara.py.

Medido el 10-oct-2026 (15 s por caso):
    hilos  receptor en el proceso      puente_camara.py
      0    3.1 fps, 321 ms             3.1 fps, 321 ms
      4    2.7 fps, 2097 ms (mÃ¡x 3.1 s) 3.1 fps, 381 ms (mÃ¡x 0.6 s)
      8    0.7 fps, 5487 ms (mÃ¡x 10 s)  2.9 fps, 395 ms (mÃ¡x 0.7 s)
Los ~320 ms de base son el tiempo de pasar 79 kB por el WiFi.

Uso: py probar_camara_carga_offline.py
"""
import json
import multiprocessing as mp
import os
import queue
import socket
import struct
import subprocess
import sys
import threading
import time

ANCHO, ALTO = 324, 244
TAM = ANCHO * ALTO
PAYLOAD = 1018          # bytes de datos por paquete CPX
KBS = 240.0             # caudal del WiFi del AI-deck medido en vuelo


def servidor(puerto, enviados, dur, modo):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('127.0.0.1', puerto))
    s.listen(1)
    c, _ = s.accept()
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    if modo == 'mudo':                 # GAP8 colgado: socket abierto, sin datos
        time.sleep(dur)
        c.close(); s.close()
        return
    img = bytes((i * 7) & 0xFF for i in range(TAM))
    t_fin = time.perf_counter() + (1.5 if modo == 'corte' else dur)
    try:
        while time.perf_counter() < t_fin:
            t0 = time.perf_counter()
            enviados.value += 1
            img = struct.pack('<d', t0) + img[8:]
            cab = struct.pack('<BHHBBI', 0xBC, ANCHO, ALTO, 1, 0, TAM)
            c.sendall(struct.pack('<HBB', len(cab) + 2, 0, 0) + cab)
            n = 0
            for k in range(0, TAM, PAYLOAD):
                trozo = img[k:k + PAYLOAD]
                c.sendall(struct.pack('<HBB', len(trozo) + 2, 0, 0) + trozo)
                n += len(trozo) + 4
                objetivo = t0 + n / (KBS * 1024)
                while time.perf_counter() < objetivo:
                    time.sleep(0.0005)
    except OSError:
        pass
    c.close()
    s.close()


def _carga(stop):
    x = 0
    while not stop.is_set():
        for i in range(2000):
            x += i * i
        time.sleep(0)


def escenario(receptor, n_hilos, dur, modo_srv='normal', puerto=5650):
    """Corre en un proceso propio; imprime una lÃ­nea JSON con el resultado."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import Prueba_ruta_vision as P
    P.VISION_FPS_MINIMO = 0
    P.VISION_RECONECTAR = False
    P.VISION_TIMEOUT_LECTURA = 1.5
    P.VISION_RECEPTOR_PROCESO = (receptor == 'proceso')
    enviados = mp.Value('i', 0)
    pr = mp.Process(target=servidor, args=(puerto, enviados, dur, modo_srv), daemon=True)
    pr.start()
    time.sleep(1.0)
    cola = queue.Queue(maxsize=1)
    rx = P.HiloReceptorAIDeck('127.0.0.1', puerto, cola, bayer=False,
                              patron='BG', demosaico=0)
    stop = threading.Event()
    for _ in range(n_hilos):
        threading.Thread(target=_carga, args=(stop,), daemon=True).start()
    rx.start()
    retrasos, frames = [], 0
    t_ini = time.perf_counter()
    while time.perf_counter() - t_ini < dur - 0.5:
        try:
            fr = cola.get(timeout=0.5)
        except queue.Empty:
            continue
        frames += 1
        t_env = struct.unpack('<d', fr.imagen[0, :8, 0].tobytes())[0]
        retrasos.append(time.perf_counter() - t_env)
    el = time.perf_counter() - t_ini
    stop.set()
    rx.detener()
    rx.join(timeout=3.0)
    huerfano = rx.proc is not None and rx.proc.poll() is None
    r = sorted(retrasos)
    p = lambda q: (r[min(len(r) - 1, int(q * len(r)))] * 1000) if r else None
    print('RESULTADO ' + json.dumps(dict(
        enviados_fps=enviados.value / el, recibidos_fps=frames / el,
        p50=p(0.5), p95=p(0.95), maximo=(r[-1] * 1000 if r else None),
        caidas=rx.caidas, atascos=rx.atascos, vivo=rx.is_alive(),
        huerfano=huerfano)), flush=True)
    os._exit(0)


def correr(receptor, n_hilos, dur=10, modo_srv='normal', puerto=5650):
    salida = subprocess.run(
        [sys.executable, os.path.abspath(__file__), '--escenario', receptor,
         str(n_hilos), str(dur), modo_srv, str(puerto)],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=dur + 60)
    for linea in salida.stdout.splitlines():
        if linea.startswith('RESULTADO '):
            return json.loads(linea[len('RESULTADO '):])
    raise RuntimeError(f'sin resultado:\n{salida.stdout[-800:]}\n{salida.stderr[-800:]}')


def main():
    from banco_headless import Banco
    b = Banco()

    r = correr('proceso', 0, puerto=5651)
    b.comprobar('puente, sin carga: llegan todos los frames',
                r['recibidos_fps'] >= 0.9 * r['enviados_fps'], '>= 90 % de los enviados',
                f"{r['recibidos_fps']:.1f} de {r['enviados_fps']:.1f} fps")
    b.comprobar('puente, sin carga: retraso = solo el enlace',
                r['p95'] is not None and r['p95'] < 450, 'p95 < 450 ms', r['p95'])

    r = correr('proceso', 4, puerto=5652)
    b.comprobar('puente, 4 hilos ocupados: no pierde frames',
                r['recibidos_fps'] >= 0.9 * r['enviados_fps'], '>= 90 % de los enviados',
                f"{r['recibidos_fps']:.1f} de {r['enviados_fps']:.1f} fps")
    b.comprobar('puente, 4 hilos ocupados: no se atasca',
                r['p95'] is not None and r['p95'] < 900, 'p95 < 900 ms', r['p95'])
    b.comprobar('puente: al detener no queda el proceso vivo',
                not r['huerfano'] and not r['vivo'], 'sin huÃ©rfano', r['huerfano'])

    a = correr('actual', 4, puerto=5653)
    b.comprobar('referencia: el receptor en el proceso SÃ se atasca con carga',
                a['p95'] is not None and a['p95'] > r['p95'] * 2,
                'p95 al menos el doble que con el puente', (a['p95'], r['p95']))

    m = correr('proceso', 0, dur=6, modo_srv='mudo', puerto=5654)
    b.comprobar('puente: GAP8 mudo â†’ se registra como atasco',
                m['atascos'] >= 1, 'atascos >= 1', m['atascos'])

    c = correr('proceso', 0, dur=6, modo_srv='corte', puerto=5655)
    b.comprobar('puente: el AI-deck corta â†’ se registra como caÃ­da',
                c['caidas'] >= 1 and c['recibidos_fps'] > 0, 'caidas >= 1 con frames antes',
                (c['caidas'], c['recibidos_fps']))
    b.terminar()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--escenario':
        escenario(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]),
                  sys.argv[5], int(sys.argv[6]))
    else:
        main()
