"""
Puente de la cámara del AI-deck en un PROCESO APARTE.

Por qué existe: en vuelo, el mismo proceso de Python corre la radio del
Crazyflie, el MoCap por MQTT, el lazo de control y la evasión. Python solo
deja avanzar un hilo a la vez (GIL), y el receptor de la cámara hacía ~160
lecturas del socket por imagen: cada una esperaba su turno. Medido en el
banco (probar_camara_carga_offline.py, sin hardware): con 8 hilos ocupados
llegaban 0.7 fps de 3.1 y con hasta 4 s de retraso. En la calibración, sin
esa carga, iba fluida.

Este proceso tiene su propio GIL: lee el socket del AI-deck sin competir con
nadie y le pasa al proceso principal, por stdout, solo imágenes COMPLETAS
(una lectura por imagen allí). Siempre manda la más reciente: si el proceso
principal se atrasa, las intermedias se descartan aquí, sin dejar de vaciar
el socket (si no, se llena la ventana TCP y el GAP8 se bloquea).

Protocolo hacia el proceso principal, por imagen:
    CABECERA (<IHHBdQ): tam, ancho, alto, formato, t_captura, bytes_red
    seguido de `tam` bytes del cuerpo (RAW o JPEG, sin decodificar).
`formato` = MARCA_CONECTADO (255) y tam = 0 avisa de que ya conectó.
t_captura es time.perf_counter() al llegar la cabecera (en Windows es el
mismo reloj en todos los procesos). bytes_red es el acumulado de la sesión.

Salida: código 2 = atasco (sin datos en el plazo), 3 = caída del enlace; el
motivo va en una línea por stderr.

Uso (lo lanza Prueba_ruta_vision.py):
    python puente_camara.py <ip> <puerto> <timeout_lectura_s>
"""
import socket
import struct
import sys
import threading
import time

MAGIC_CPX = 0xBC
CABECERA = struct.Struct('<IHHBdQ')
MARCA_CONECTADO = 255


class _Lector:
    """Sirve bytes exactos desde un buffer que se llena en bloques grandes."""

    def __init__(self, sock):
        self.sock = sock
        self.buf = bytearray()
        self.pos = 0
        self.bytes_red = 0

    def exacto(self, n):
        while len(self.buf) - self.pos < n:
            trozo = self.sock.recv(1 << 17)
            if not trozo:
                raise ConnectionResetError('El AI-deck cerró la conexión')
            self.bytes_red += len(trozo)
            if self.pos > (1 << 18):
                del self.buf[:self.pos]
                self.pos = 0
            self.buf.extend(trozo)
        out = bytes(self.buf[self.pos:self.pos + n])
        self.pos += n
        return out

    def paquete(self):
        length, _dst, _src = struct.unpack('<HBB', self.exacto(4))
        return self.exacto(length - 2)


def main():
    ip, puerto, t_lectura = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    salida = sys.stdout.buffer

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect((ip, puerto))
    sock.settimeout(t_lectura)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    except OSError:
        pass

    salida.write(CABECERA.pack(0, 0, 0, MARCA_CONECTADO, time.perf_counter(), 0))
    salida.flush()

    # Una sola ranura: el escritor siempre manda la imagen más reciente.
    ranura = [None]
    cerrojo = threading.Lock()
    hay_nueva = threading.Event()
    padre_fuera = threading.Event()

    def escritor():
        while not padre_fuera.is_set():
            if not hay_nueva.wait(0.5):
                continue
            hay_nueva.clear()
            with cerrojo:
                item, ranura[0] = ranura[0], None
            if item is None:
                continue
            try:
                salida.write(item[0])
                salida.write(item[1])
                salida.flush()
            except (BrokenPipeError, OSError, ValueError):
                padre_fuera.set()

    threading.Thread(target=escritor, daemon=True, name='escritor').start()

    lector = _Lector(sock)
    while not padre_fuera.is_set():
        cab = lector.paquete()
        if len(cab) < 11:
            continue
        magic, ancho, alto, _prof, formato, tam = struct.unpack('<BHHBBI', cab[:11])
        if magic != MAGIC_CPX:
            continue
        t_captura = time.perf_counter()
        cuerpo = bytearray()
        while len(cuerpo) < tam:
            cuerpo.extend(lector.paquete())
        item = (CABECERA.pack(len(cuerpo), ancho, alto, formato, t_captura,
                              lector.bytes_red), bytes(cuerpo))
        with cerrojo:
            ranura[0] = item
        hay_nueva.set()
    sock.close()


if __name__ == '__main__':
    try:
        main()
    except socket.timeout:
        print(f'ATASCO sin datos en {float(sys.argv[3]):.0f}s (GAP8 atascado: '
              'el socket sigue abierto pero no llegan frames)',
              file=sys.stderr, flush=True)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f'CAIDA {type(e).__name__}: {e}', file=sys.stderr, flush=True)
        sys.exit(3)
