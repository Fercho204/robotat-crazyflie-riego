"""Banco nuevo; no reemplaza los tres bancos ausentes de esta copia."""
import numpy as np
from banco_headless import Banco
from evasion import desvio_lateral, MapaObstaculos, hay_que_parar


def main():
    b = Banco()
    for y in (0., -.1, .1):
        d, _, stop = desvio_lateral((0, 0), (1, 0), [[.1, y]], [1],
                                    holgura=.43, desvio_max=.6)
        b.comprobar(f'holgura y={y}', abs(abs(d-y)-.43)<1e-9 and not stop,
                    0, round(abs(d-y), 3))
    ds = [desvio_lateral((0, 0), (1, 0), [[.1, y]], [1], lado=1)[0]
          for y in np.linspace(-.1, .1, 41)]
    cambios = sum(a*c < 0 for a, c in zip(ds, ds[1:]))
    b.comprobar('compromiso', cambios == 0, 0, cambios)
    m = MapaObstaculos(vida_media=1)
    m.actualizar([[.1, 0]], 0)
    m.decaer(10)
    b.comprobar('olvido', len(m.pts)==0, 1, len(m.pts))
    stop, _ = hay_que_parar([[.1, 0]], (0, 0), (1, 0))
    b.comprobar('parada central', stop, 0, int(stop))
    return b.terminar()


if __name__ == '__main__':
    raise SystemExit(main())
