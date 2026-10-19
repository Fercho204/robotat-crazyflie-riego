"""Muestreo híbrido puro con mapas sintéticos, sin modelos ni hardware."""
import numpy as np
from banco_headless import Banco
from evasion import MapaObstaculos, detectar_columnas_vertical, valor_por_columna


def main():
    b=Banco()
    det=np.ones((60,94),dtype=float)
    val=np.arange(60*94,dtype=float).reshape(60,94)
    # Bordes no uniformes: columna 4 (índice cero), un poste de un píxel.
    det[12:48,44]=100.
    val[12:48,44]=37.
    sup,_=detectar_columnas_vertical(det,percentil=97.)
    resultado=valor_por_columna(det,val,sup)
    b.comprobar('solo pixeles del objeto en columna 4',resultado[4]==37.,37.,resultado[4])
    bordes=np.linspace(0,94,10).astype(int)
    esperados=np.array([np.median(val[12:48,bordes[i]:bordes[i+1]]) for i in range(9)])
    sin_objeto=np.arange(9)!=4
    b.comprobar('columnas sin objeto usan mediana de franja',
                np.allclose(resultado[sin_objeto],esperados[sin_objeto]),
                esperados[sin_objeto],resultado[sin_objeto])
    vacio=valor_por_columna(det,val,np.full(9,101.))
    b.comprobar('mascara vacia usa toda la franja',np.allclose(vacio,esperados),esperados,vacio)
    det[:12]=1000.; det[48:]=1000.
    val[:12]=-999.; val[48:]=-999.
    fuera=valor_por_columna(det,val,sup)
    b.comprobar('pixeles fuera de franja no cuentan',np.array_equal(fuera,resultado),resultado,fuera)
    b.comprobar('salida float de nueve columnas',resultado.shape==(9,) and resultado.dtype.kind=='f',
                (9,'float'),(resultado.shape,resultado.dtype))
    # Olvido continuo con tiempo simulado: cada avistamiento suma uno.
    punto = np.array([[1.0, 0.5]])
    mapa = MapaObstaculos(vida_media=2.0)
    mapa.actualizar(punto, 10.0)
    pts, pesos = mapa.confirmados(1.5)
    b.comprobar('un avistamiento no confirma con 1.5',
                len(pts) == 0 and len(pesos) == 0, 0, len(pts))
    mapa.actualizar(punto, 10.5)
    pts, pesos = mapa.confirmados(1.5)
    peso_esperado = 1.0 + 0.5 ** (0.5 / 2.0)
    b.comprobar('dos avistamientos a 0.5 s confirman con 1.5',
                pts.shape == punto.shape and np.allclose(pts, punto)
                and pesos.shape == (1,) and np.allclose(pesos, [peso_esperado]),
                (punto, [peso_esperado]), (pts, pesos))
    pts, pesos = mapa.confirmados(2.0)
    b.comprobar('dos avistamientos a 0.5 s no confirman con 2.0',
                len(pts) == 0 and len(pesos) == 0, 0, len(pts))
    mapa = MapaObstaculos(vida_media=2.0)
    mapa.actualizar(punto, 10.0)
    mapa.actualizar(punto, 13.0)
    pts, pesos = mapa.confirmados(1.5)
    b.comprobar('dos avistamientos a 3 s no confirman con 1.5',
                len(pts) == 0 and len(pesos) == 0, 0, len(pts))
    return b.terminar()


if __name__=='__main__': raise SystemExit(main())
