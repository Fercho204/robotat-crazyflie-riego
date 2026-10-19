"""Estado de batería aislado: extrae _avisar sin importar hardware."""
from types import SimpleNamespace as NS
from banco_headless import Banco, extraer_metodo


def main():
    b = Banco()
    reloj = [100.0]
    estado = {'v': None, 't': 0.0, 'critica_desde': None}
    avisar = extraer_metodo('Prueba_ruta_vision.py', 'MonitorBateria', '_avisar',
                           {'time': NS(time=lambda: reloj[0]),
                            '_bateria_estado': estado,
                            'BATERIA_CRITICA_V': 3.0, 'BATERIA_AVISO_V': 3.3})
    monitor = NS(_t_aviso=0.0, _aviso_dado=False)
    avisar(monitor, 3.2)
    b.comprobar('3.2 V sin estado critico',
                estado == {'v': 3.2, 't': 100.0, 'critica_desde': None},
                'lectura registrada, sin critica', estado.copy())
    reloj[0] = 101.0
    avisar(monitor, 2.95)
    inicio = estado['critica_desde']
    b.comprobar('2.95 V inicia critica',
                estado == {'v': 2.95, 't': 101.0, 'critica_desde': 101.0},
                101.0, estado.copy())
    reloj[0] = 103.0
    avisar(monitor, 2.9)
    b.comprobar('lectura critica conserva el inicio',
                estado == {'v': 2.9, 't': 103.0, 'critica_desde': inicio},
                inicio, estado.copy())
    reloj[0] = 104.0
    avisar(monitor, 3.1)
    b.comprobar('3.1 V borra critica',
                estado == {'v': 3.1, 't': 104.0, 'critica_desde': None},
                None, estado.copy())
    anterior = estado.copy()
    reloj[0] = 105.0
    avisar(monitor, 0.0)
    b.comprobar('0.0 V no toca el estado', estado == anterior,
                anterior, estado.copy())
    # Una lectura inválida tampoco debe borrar una crítica en curso.
    avisar(monitor, 2.95)
    anterior = estado.copy()
    reloj[0] = 106.0
    avisar(monitor, 0.0)
    b.comprobar('0.0 V conserva critica en curso', estado == anterior,
                anterior, estado.copy())
    avisar(monitor, 3.0)
    b.comprobar('umbral exacto borra critica', estado['critica_desde'] is None,
                None, estado.copy())
    return b.terminar()


if __name__ == '__main__':
    raise SystemExit(main())
