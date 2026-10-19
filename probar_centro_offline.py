import numpy as np
from banco_headless import Banco
from evasion import desvio_lateral, salida_lateral_segura, hay_que_parar


def main():
    b = Banco()
    lados = []
    antes_lados = []
    for y in (-.001, 0, .001):
        izq, der = y+.43, y-.43
        bruto = izq if abs(izq) <= abs(der) else der
        if abs(abs(izq)-abs(der)) < 1e-3:
            bruto = der if y > 0 else izq
        antes_lados.append(int(np.sign(bruto)))
        d, _, _ = desvio_lateral((0,0), (1,0), [[.2,y]], [1], holgura=.43)
        lados.append(np.sign(d))
    b.comprobar('margen central estable', lados == [1,1,1], antes_lados, lados)
    pos = np.array([0.,0.])
    obst = np.array([[.2,0.]])
    # Antes: repetir pos mantiene la parada indefinidamente.
    antes = sum(hay_que_parar(obst, pos, (1,0))[0] for _ in range(100))
    n = 0
    for _ in range(100):
        if not hay_que_parar(obst, pos, (1,0))[0]: break
        candidato = salida_lateral_segura(pos, pos+[0,.005], (1,0), obst)
        if candidato is not None: pos = candidato
        n += 1
    b.comprobar('salida del bloqueo frontal', n < 100 and pos[0] == 0, antes, n)
    veto = salida_lateral_segura((0,0), (0,.5), (1,0), [[.2,0],[0,.3]])
    b.comprobar('obstaculo lateral veta salida', veto is None, 0, int(veto is None))
    ds = [desvio_lateral((0,0),(1,0),[[.2,y]],[1],lado=1)[0]
          for y in np.linspace(-.1,.1,100)]
    cambios = sum(a*c < 0 for a,c in zip(ds,ds[1:]))
    b.comprobar('lado comprometido', cambios == 0, 0, cambios)
    return b.terminar()


if __name__ == '__main__': raise SystemExit(main())
