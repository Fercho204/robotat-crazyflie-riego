"""Contrato del adaptador con dobles NumPy; NO mide inferencia neuronal real."""
import contextlib
import os
import sys
import time
from types import SimpleNamespace as NS
import numpy as np
from banco_headless import Banco, extraer
from evasion import detectar_columnas_vertical


class Tensor:
    def __init__(self, x): self.x = np.asarray(x)
    def to(self, *a): return self
    def eval(self): return self
    def unsqueeze(self, n): return Tensor(np.expand_dims(self.x, n))
    def __getitem__(self, key): return Tensor(self.x[key])
    def cpu(self): return self
    def numpy(self): return self.x


def main():
    b = Banco()
    cargas = []
    imagen = np.zeros((60, 90, 3), dtype=np.uint8)
    imagen[:, 40:50] = 200
    def entrada(img): return Tensor(img[:, :, 0][None].astype(float)+1)
    class Modelo:
        def to(self, *a): return self
        def eval(self): return self
        def __call__(self, x=None, **kw):
            return x if x is not None else NS(predicted_depth=kw['pixel_values'])
    def cargar(repo, nombre, **kw):
        cargas.append(nombre)
        return NS(small_transform=entrada, dpt_transform=entrada) if nombre == 'transforms' else Modelo()
    red = []
    processor_kwargs = []
    cache_vacia = [False]
    def _cargar_hf(nombre, kw):
        if kw.get('local_files_only'):
            if cache_vacia[0]: raise OSError('no está en caché')
        else:
            red.append(nombre)
    class Processor:
        @classmethod
        def from_pretrained(cls, nombre, **kw):
            _cargar_hf(nombre, kw)
            cargas.append('processor'); return cls()
        def __call__(self, images, **kw):
            processor_kwargs.append(kw.copy())
            return {'pixel_values': entrada(images)}
    class Auto:
        @staticmethod
        def from_pretrained(nombre, **kw):
            _cargar_hf(nombre, kw); cargas.append(nombre); return Modelo()
    sys.modules['transformers'] = NS(AutoImageProcessor=Processor, AutoModelForDepthEstimation=Auto)
    torch = NS(set_num_threads=lambda n: None, device=lambda x:x,
               hub=NS(load=cargar), no_grad=contextlib.nullcontext,
               nn=NS(functional=NS(interpolate=lambda x, **kw:x)))
    scope = extraer('Prueba_ruta_vision.py', ['EstimadorMiDaS'],
                    dict(torch=torch, _HAY_TORCH=True, VISION_HILOS_TORCH=1,
                         os=os, cv2=NS(COLOR_BGR2RGB=1, cvtColor=lambda x,c:x[:,:,::-1])))
    for nombre in ('midas', 'depth_anything_v2'):
        e = scope['EstimadorMiDaS'](nombre, 'cpu')
        n = len(cargas)
        t0 = time.perf_counter()
        for _ in range(10): mapa = e.inferir(imagen)
        ms = (time.perf_counter()-t0)*100
        b.comprobar(nombre+' forma/tipo', mapa.shape==(60,90) and mapa.dtype==np.float32,
                    imagen.shape, (mapa.shape,str(mapa.dtype)))
        b.comprobar(nombre+' carga unica', len(cargas)==n, n, len(cargas))
        disp, ratio = detectar_columnas_vertical(mapa)
        b.comprobar(nombre+' columna central', np.argmax(disp)==4 and ratio[4]==1,
                    0, float(disp[4]))
        print(f'{nombre}: doble de prueba {ms:.3f} ms; NO es tiempo de inferencia real')
    # Tamaño opcional del procesador de Depth Anything.
    e336 = scope['EstimadorMiDaS']('depth_anything_v2', 'cpu', lado_entrada=336)
    e336.inferir(imagen)
    esperado_size = {'height': 336, 'width': 336}
    b.comprobar('depth_anything_v2 lado 336 pasa size',
                processor_kwargs[-1].get('size') == esperado_size,
                esperado_size, processor_kwargs[-1])
    e_default = scope['EstimadorMiDaS']('depth_anything_v2', 'cpu')
    e_default.inferir(imagen)
    b.comprobar('depth_anything_v2 por defecto no pasa size',
                'size' not in processor_kwargs[-1],
                'sin size', processor_kwargs[-1])
    # Variante métrica: el modelo da metros y el estimador los entrega como 1/m
    # (alto = cerca), para que el detector y la calibración a=1, b=0 valgan igual.
    em = scope['EstimadorMiDaS']('depth_anything_v2_metrico', 'cpu')
    mm = em.inferir(imagen)
    esperado = 1.0 / np.maximum(imagen[:, :, 0].astype(np.float64) + 1, 0.1)
    b.comprobar('metrico entrega 1/m float32', mm.dtype == np.float32
                and mm.shape == (60, 90) and np.allclose(mm, esperado, atol=1e-6),
                'float32 (60,90)', (str(mm.dtype), mm.shape))
    b.comprobar('metrico usa su propio modelo',
                'Metric-Indoor' in em.ID_DEPTH_ANYTHING_METRICO and em.es_metrico,
                True, em.es_metrico)
    # Sin internet en vuelo: con la caché llena no debe tocar la red; con la
    # caché vacía debe reintentar con red en vez de fallar.
    b.comprobar('depth_anything_v2 con cache no usa red', red == [], [], red)
    cache_vacia[0] = True
    e = scope['EstimadorMiDaS']('depth_anything_v2', 'cpu')
    b.comprobar('depth_anything_v2 sin cache reintenta con red',
                len(red) == 2 and e.inferir(imagen).shape == (60, 90), 2, len(red))
    print('LIMITACION: dobles NumPy; la latencia real se midio aparte (ver LEEME).')
    return b.terminar()


if __name__ == '__main__':
    raise SystemExit(main())
