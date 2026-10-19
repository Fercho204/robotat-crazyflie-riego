"""Prueba Evasion real por AST, sin importar cflib, cámara ni OptiTrack."""
from types import SimpleNamespace
import threading
import time
import numpy as np
import evasion
from banco_headless import Banco, extraer, extraer_metodo
from probar_maniobra_offline import constantes


def main():
    b=Banco()
    cfg=constantes()
    # constantes() solo lee literales; EVASION_PERCENTIL se deriva del modelo.
    cfg['EVASION_PERCENTIL']=cfg['EVASION_PERCENTIL_POR_MODELO'].get(cfg['VISION_MODELO'],90.0)
    cfg.update(EVASION_MODO='esquivar', EVASION_DISP_AUTO=False, EVASION_DISP_MIN=10.)
    scope=extraer('Prueba_ruta_carrot.py',['Evasion','_MidiendoFondo'],cfg)
    scope['VISION_MODELO']='depth_anything_v2_metrico'
    e_metrico=scope['Evasion'](SimpleNamespace())
    b.comprobar('modelo metrico fuerza observar activo',not e_metrico.esquiva and e_metrico.activa,
                (True,True),(e_metrico.esquiva,e_metrico.activa))
    scope['VISION_MODELO']='midas'
    e_midas=scope['Evasion'](SimpleNamespace())
    b.comprobar('MiDaS permite esquivar',e_midas.esquiva,True,e_midas.esquiva)
    imagen=np.tile(np.repeat(np.array([10,11,12,11,100,12,11,10,12.]),10),(60,1)).astype('float32')
    u={'mapa':imagen, 'mapa_ts':1., 'ts':1.}
    e=scope['Evasion'](SimpleNamespace(ultimo=lambda:dict(u)))
    e._ev=evasion
    e.mapa=evasion.MapaObstaculos()
    e.cal=evasion.CalibracionMiDaS()
    e.rumbos=evasion.rumbos_columnas(cfg['EVASION_N_COL'],cfg['EVASION_FOV_DEG'])
    e.paso((0,0),0,(.4,0),(1,0),1.)
    n=e.n_det
    b.comprobar('mapa sintetico detectado',n>0 and e.vista['bloqueada'],0,n)
    u['ts']=2.
    e.paso((0,0),0,(.4,0),(1,0),2.)
    b.comprobar('video nuevo no renueva profundidad',e.n_det==n and e.vista['ts']==1.,n,e.n_det)
    e.lado_comp=1; e.t_comp=0.; e.mantener_compromiso=True
    e._soltar_lado()
    e.paso((0,0),0,(.4,0),(1,0),100.)
    b.comprobar('olvido y timeout no sueltan compromiso',e.lado_comp==1,1,e.lado_comp)
    e.mantener_compromiso=False; e._soltar_lado()
    b.comprobar('fin libera compromiso',e.lado_comp==0,1,e.lado_comp)
    u.update(mapa=np.full_like(imagen,np.nan),mapa_ts=101.)
    e.paso((0,0),0,(.4,0),(1,0),101.)
    b.comprobar('NaN nunca significa libre',not e.vista['valida'],1,int(e.vista['valida']))
    scope['VISION_MODELO']='depth_anything_v2'
    e.disp_fondo=None
    b.comprobar('compuerta MiDaS no cruza modelo',e.compuerta is None,10,e.compuerta)
    scope['VISION_MODELO']='hibrido'
    scope['EVASION_PERCENTIL']=cfg['EVASION_PERCENTIL_POR_MODELO']['hibrido']
    e.disp_fondo=10.
    e.cal=evasion.CalibracionMiDaS(.01,1.)
    # MiDaS=25 -> 0.8 m; detección=100 -> 0.5 m con el mismo ajuste.
    for etiqueta,distancia,esperada in (
            ('hibrido',np.full_like(imagen,25.),.8),
            ('forma invalida',np.ones((2,2)),.5),
            ('no finito',np.full_like(imagen,np.nan),.5),
            ('sin mapa_dist',None,.5)):
        e.mapa=evasion.MapaObstaculos()
        u.update(mapa=imagen,mapa_dist=distancia,mapa_ts=u['mapa_ts']+1.)
        e.paso((0,0),0,(.4,0),(1,0),u['mapa_ts'])
        pts=e.mapa.pts
        b.comprobar('distancia proyectada '+etiqueta,
                    pts.shape==(1,2) and np.allclose(pts,[[esperada,0.]]),
                    [[esperada,0.]],pts)
    # Si se fija izquierda, no se permite usar la viabilidad derecha para avanzar.
    _,_,stop=evasion.desvio_lateral((0,0),(1,0),[[.1,.2]],[1],
                                    lado=1,holgura=.43,desvio_max=.6)
    b.comprobar('no_cabe respeta lado comprometido',stop,False,stop)
    d,_,stop=evasion.desvio_lateral((0,0),(1,0),[[.1,.01]],[1],
                                   holgura=.60,desvio_max=.60)
    b.comprobar('preferencia no elige lado inviable',d<0 and not stop,.61,d)
    escape=evasion.salida_lateral_segura((0,0),(0,.43),(1,0),[[.001,.02]],radio=.43)
    b.comprobar('retroceso excesivo se veta',escape is None,8.6,escape)
    detener=threading.Event()
    vision=SimpleNamespace(_detener=detener, _lock_midas=threading.Lock(),
                           _lock=threading.Lock(), _entrada_midas=(imagen,123.),
                           _ultimo={}, estimador_dist=None, ms_det=0.0, ms_dist=0.0)
    llamadas=[]
    class Modelo:
        def __init__(self,*args,**kw): llamadas.append((args,kw))
        def inferir(self,img): detener.set(); return img
    worker=extraer_metodo('Prueba_ruta_vision.py','SistemaVision','_bucle_midas',
                          dict(_bajar_prioridad_hilo=lambda:None,VISION_MIDAS=True,
                               _HAY_TORCH=True,EstimadorMiDaS=Modelo,VISION_MODELO='depth_anything_v2',
                               VISION_DISPOSITIVO='cpu',VISION_DA_LADO=336,VISION_MIN_PERIODO=.2,time=time,
                               stop_event=threading.Event(),analizar_obstaculos=lambda m:{}))
    worker(vision)
    b.comprobar('worker selecciona, carga una vez y pasa lado 336',
                llamadas==[(('depth_anything_v2','cpu'),{'lado_entrada':336})],1,llamadas)
    b.comprobar('worker conserva timestamp de entrada',vision._ultimo['mapa_ts']==123.,123.,vision._ultimo['mapa_ts'])
    b.comprobar('worker simple no publica distancia',vision._ultimo['mapa_dist'] is None,
                None,vision._ultimo['mapa_dist'])
    detener.clear(); llamadas.clear()
    entradas=[]
    class ModeloHibrido:
        def __init__(self,*args,**kw):
            llamadas.append((args,kw)); self.nombre=args[0]
        def inferir(self,img):
            entradas.append(img)
            if self.nombre=='midas':
                detener.set()
                return np.full_like(img,25.)
            return img
    vision._entrada_midas=(imagen,124.)
    worker=extraer_metodo('Prueba_ruta_vision.py','SistemaVision','_bucle_midas',
                          dict(_bajar_prioridad_hilo=lambda:None,VISION_MIDAS=True,
                               _HAY_TORCH=True,EstimadorMiDaS=ModeloHibrido,VISION_MODELO='hibrido',
                               VISION_DISPOSITIVO='cpu',VISION_DA_LADO=336,VISION_MIN_PERIODO=.2,time=time,
                               stop_event=threading.Event(),analizar_obstaculos=lambda m:{}))
    worker(vision)
    esperado=[(('depth_anything_v2','cpu'),{'lado_entrada':336}),(('midas','cpu'),{})]
    b.comprobar('worker hibrido crea ambos modelos una vez',llamadas==esperado,esperado,llamadas)
    b.comprobar('worker hibrido infiere la misma imagen',
                len(entradas)==2 and all(img is imagen for img in entradas),2,len(entradas))
    b.comprobar('worker hibrido publica ambos mapas y timestamp',
                vision._ultimo['mapa'] is imagen and vision._ultimo['mapa_ts']==124.
                and vision._ultimo['mapa_dist'] is vision._mapa_dist
                and np.all(vision._mapa_dist==25.),124.,vision._ultimo['mapa_ts'])
    b.comprobar('worker hibrido mide el tiempo de cada modelo',
                vision.ms_det>0 and vision.ms_dist>0,'>0 ms',(vision.ms_det,vision.ms_dist))
    return b.terminar()


if __name__=='__main__': raise SystemExit(main())
