"""
evasion.py — Detección de obstáculos con MiDaS y desvío de la zanahoria.

Se usa desde volar_ruta.py. Todo lo de aquí son funciones puras
o clases sin estado global, para poder probarlo sin volar.

══════════════════════════════════════════════════════════════════════════
LO QUE HAY QUE SABER ANTES DE FIARSE DE ESTO

1. MiDaS NO DA METROS.
   Da profundidad inversa relativa (disparidad): valor alto = cerca. Y
   está entrenado con una pérdida invariante a escala y desplazamiento,
   así que el mapa solo está definido salvo una transformación afín que
   PUEDE CAMBIAR DE UNA IMAGEN A OTRA.
   Un ajuste global  d[m] = 1/(a·disp + b)  funciona razonablemente con
   una cámara fija y escenas parecidas, pero es aproximado y se degrada
   cuando la escena cambia mucho. Por eso aquí NO se decide nada crítico
   solo con la distancia absoluta: la señal principal es RELATIVA (esta
   columna está mucho más cerca que el resto del cuadro), y la distancia
   calibrada solo gradúa la urgencia.

2. LA CÁMARA MIRA A DONDE MIRA LA NARIZ.
   Con YAW_MODO='perpendicular' el eje óptico va a 90° del avance, así que
   un obstáculo JUSTO DELANTE queda fuera del encuadre (con FOV de 70°, el
   frente cae 55° fuera del borde). Con esa configuración NO hay protección
   frontal, y este módulo lo dice en voz alta.
   Lo que sí se puede en perpendicular: mapear lo que hay AL LADO y
   esquivarlo cuando la ruta pase por ahí — en una ruta de cobertura ves
   el surco de al lado un pase antes.
   Para protección frontal: YAW_MODO = 'tangente'.

3. TODO SE PROYECTA A COORDENADAS DEL MUNDO.
   Una detección se convierte en un punto (x, y) del laboratorio usando la
   posición del dron y hacia dónde apunta la cámara EN ESE INSTANTE. Así la
   evasión es geometría pura y no depende de cómo esté orientada la nariz.
"""

import numpy as np


# ══════════════════════════════════════════════════════════════
# CALIBRACIÓN  disparidad de MiDaS  →  metros
# ══════════════════════════════════════════════════════════════

class CalibracionMiDaS:
    """
    Modelo  distancia = 1 / (a·disp + b),  ajustado por mínimos cuadrados
    sobre pares (disparidad medida, distancia real medida con cinta).

    Es la forma correcta: MiDaS da disparidad, y disparidad ∝ 1/distancia.
    Ajustar una recta a (disp, distancia) sería el modelo equivocado y se
    dispararía en el rango lejano.
    """

    def __init__(self, a=None, b=None):
        self.a, self.b = a, b

    @property
    def valida(self):
        return self.a is not None and self.b is not None

    def ajustar(self, disparidades, distancias):
        d = np.asarray(disparidades, float)
        z = np.asarray(distancias, float)
        if len(d) < 3:
            raise ValueError('Hacen falta al menos 3 medidas.')
        # 1/z = a·disp + b  → recta en (disp, 1/z)
        A = np.column_stack([d, np.ones_like(d)])
        sol, *_ = np.linalg.lstsq(A, 1.0 / z, rcond=None)
        self.a, self.b = float(sol[0]), float(sol[1])
        pred = self.metros(d)
        err = np.abs(pred - z)
        return {'a': self.a, 'b': self.b,
                'err_medio': float(err.mean()),
                'err_max': float(err.max()),
                'r2': float(1.0 - np.sum((1/z - A @ sol) ** 2)
                            / max(np.sum((1/z - np.mean(1/z)) ** 2), 1e-12))}

    def metros(self, disp):
        """Distancia en metros. inf donde el modelo no tiene sentido."""
        if not self.valida:
            return np.full(np.shape(disp), np.inf, dtype=float)
        inv = self.a * np.asarray(disp, float) + self.b
        with np.errstate(divide='ignore', invalid='ignore'):
            z = np.where(inv > 1e-6, 1.0 / np.maximum(inv, 1e-6), np.inf)
        return z


# ══════════════════════════════════════════════════════════════
# DETECCIÓN EN LA IMAGEN
# ══════════════════════════════════════════════════════════════

def detectar_columnas(mapa, n_col=9, franja=(0.30, 0.80), percentil=90.0):
    """
    Divide el mapa en n_col columnas y devuelve, por columna, la
    disparidad del percentil alto (lo MÁS CERCANO de esa columna).

    Se usa un percentil y no el máximo porque el máximo es un píxel suelto
    y MiDaS tiene bastante sal y pimienta en los bordes.
    """
    alto, ancho = mapa.shape
    y0, y1 = int(alto * franja[0]), int(alto * franja[1])
    banda = mapa[y0:y1, :]
    bordes = np.linspace(0, ancho, n_col + 1).astype(int)
    return np.array([np.percentile(banda[:, bordes[i]:bordes[i + 1]], percentil)
                     for i in range(n_col)], dtype=float)


def detectar_columnas_vertical(mapa, n_col=9, franja=(0.20, 0.80),
                               percentil=90.0):
    """
    Como detectar_columnas, pero además devuelve el GRADIENTE VERTICAL de
    cada columna. Es lo que separa un obstáculo del suelo.

    LA FÍSICA
    Con la cámara a altura h mirando horizontal, un píxel a ángulo θ por
    debajo de la horizontal ve el suelo a  d = h/tan(θ),  o sea que la
    disparidad del suelo es  tan(θ)/h : CRECE sin parar según bajas por la
    imagen, porque el suelo se acerca.
    Un poste vertical (un trípode, un tronco) está a la MISMA distancia en
    todas sus filas: disparidad constante.

    Medido sobre escenas sintéticas con ruido:
        suelo    → ratio inferior/superior ≈ 10
        trípode  → ratio ≈ 1.0
    Diez veces de separación. No hace falta calibrar nada para usarlo.

    Devuelve (disp_superior, ratio_inferior_superior).
    Se devuelve la disparidad de la mitad SUPERIOR a propósito: es la que
    corresponde a lo que está a la altura del dron o por encima, que es lo
    que le puede pegar.
    """
    alto, ancho = mapa.shape
    y0, y1 = int(alto * franja[0]), int(alto * franja[1])
    ym = (y0 + y1) // 2
    bordes = np.linspace(0, ancho, n_col + 1).astype(int)
    sup = np.array([np.percentile(mapa[y0:ym, bordes[i]:bordes[i + 1]], percentil)
                    for i in range(n_col)], dtype=float)
    inf = np.array([np.percentile(mapa[ym:y1, bordes[i]:bordes[i + 1]], percentil)
                    for i in range(n_col)], dtype=float)
    return sup, inf / np.maximum(sup, 1e-6)


def valor_por_columna(mapa_det, mapa_val, sup_det, n_col=9, franja=(0.20, 0.80)):
    """Depth Anything decide dónde; MiDaS mide la distancia en esos píxeles."""
    alto, ancho = mapa_det.shape
    y0, y1 = int(alto * franja[0]), int(alto * franja[1])
    bordes = np.linspace(0, ancho, n_col + 1).astype(int)
    valores = []
    for i in range(n_col):
        det = mapa_det[y0:y1, bordes[i]:bordes[i + 1]]
        val = mapa_val[y0:y1, bordes[i]:bordes[i + 1]]
        mascara = det >= sup_det[i]
        valores.append(np.median(val[mascara] if np.any(mascara) else val))
    return np.array(valores, dtype=float)


def rumbos_columnas(n_col, fov_deg):
    """Ángulo de cada columna respecto al eje óptico. + = a la izquierda
    de la imagen si el eje X de la imagen crece a la derecha."""
    centros = (np.arange(n_col) + 0.5) / n_col - 0.5     # -0.5 .. +0.5
    return -centros * np.radians(fov_deg)


def z_robusto(disp):
    """
    Cuántas desviaciones típicas destaca cada columna sobre el fondo del
    propio cuadro, con mediana y MAD en vez de media y sigma.

        z = (disp − mediana) / (1.4826 · MAD)

    Se usa mediana/MAD y no media/sigma porque el obstáculo que queremos
    detectar está DENTRO de la muestra: con media y sigma, un obstáculo
    muy marcado se infla a sí mismo el umbral y deja de destacar.

    Devuelve el vector de z. Si el cuadro es perfectamente plano (MAD=0)
    devuelve ceros: no hay nada que destaque.
    """
    d = np.asarray(disp, float)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    sigma = 1.4826 * mad
    if sigma <= 1e-9:
        return np.zeros(len(d), dtype=float)
    return (d - med) / sigma


def columnas_sospechosas(disp, umbral_rel=1.25, ratio=None, max_ratio=1.6,
                         sigmas=None, disp_min=None):
    """
    Qué columnas destacan sobre el fondo del propio cuadro.

    Señal RELATIVA, que es la fiable: no depende de la calibración.

    DOS PRUEBAS, Y LA SEGUNDA ES LA QUE HACE EL TRABAJO
    `umbral_rel` compara contra la MEDIANA: disp > umbral_rel · mediana.
    Por sí sola dispara casi siempre, y eso es un defecto de fondo, no de
    ajuste: en CUALQUIER cuadro hay alguna columna por encima de 1.25 veces
    la mediana, incluso mirando a una pared lisa, porque el ruido de MiDaS
    basta para llegar ahí. No existe el estado "aquí no hay nada".
    Medido en vuelo sobre el Robotat VACÍO: 111 detecciones proyectadas y
    748 ciclos (37% del vuelo) en que se habría apartado al tope.

    `sigmas` compara contra la DISPERSIÓN del propio cuadro: exige
    z_robusto > sigmas. En un cuadro plano y ruidoso la dispersión es
    pequeña pero el exceso también, así que el z se queda bajo y NO
    dispara. Cuando hay algo de verdad cerca, destaca por muchos sigmas.
    Esa es la diferencia entre "el máximo del cuadro" —que siempre existe—
    y "algo que se sale de lo normal del cuadro".

    `disp_min` es una tercera prueba, y es la que ataca el fallo de verdad
    observado en vuelo: una compuerta ABSOLUTA sobre la disparidad bruta.
    Las dos anteriores son relativas al cuadro, así que un objeto lejano
    pero real (una pared, la estructura del laboratorio) las pasa sin
    problema — destaca de verdad. Lo que hay que exigirle además es estar
    CERCA, y eso no se puede preguntar a una medida relativa. Se compara
    sobre el número bruto de MiDaS, antes de convertir a metros, para no
    heredar el error de la calibración. Se fija midiendo el fondo del
    propio laboratorio con un vuelo sin obstáculos.

    Cualquiera de los tres a None se salta.
    """
    d = np.asarray(disp, float)
    med = float(np.median(d))
    if med <= 1e-9:
        return np.zeros(len(d), dtype=bool)
    m = d > umbral_rel * med
    if sigmas is not None:
        m &= z_robusto(d) > float(sigmas)
    if disp_min is not None:
        m &= d > float(disp_min)
    if ratio is not None:
        m &= np.asarray(ratio, float) < max_ratio
    return m


def filtrar_por_zona(pts, ruta_xy=None, dist_max=None, zonas_prohibidas=()):
    """
    Descarta detecciones por DÓNDE CAEN en el laboratorio.

    Es el filtro que hace falta para la estructura fija del Robotat (el
    marco, las patas, el borde de la mesa): eso es un objeto real y
    vertical, así que el test de gradiente no lo quita. Lo que lo quita es
    saber que está donde no vuelas.

      ruta_xy + dist_max : solo se aceptan detecciones a menos de dist_max
                           de la ruta planificada. Cero configuración, y
                           se lleva por delante paredes, la estructura del
                           laboratorio y la gente mirando.
      zonas_prohibidas   : [(x, y, radio), ...] para excluir sitios
                           concretos a mano.

    Devuelve la máscara de los que se QUEDAN.
    """
    pts = np.asarray(pts, float).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    ok = np.ones(len(pts), dtype=bool)

    if ruta_xy is not None and dist_max is not None:
        r = np.asarray(ruta_xy, float)[:, :2]
        # distancia de cada punto al vértice más cercano de la ruta.
        # Basta con los vértices porque la ruta viene remuestreada fina.
        for i, p in enumerate(pts):
            if np.min(np.linalg.norm(r - p, axis=1)) > dist_max:
                ok[i] = False

    for (zx, zy, zr) in zonas_prohibidas:
        ok &= np.linalg.norm(pts - np.array([zx, zy]), axis=1) > zr

    return ok


# ══════════════════════════════════════════════════════════════
# PROYECCIÓN A COORDENADAS DEL MUNDO
# ══════════════════════════════════════════════════════════════

def a_mundo(pos_xy, rumbo_camara, rumbos, distancias):
    """
    Convierte detecciones (ángulo respecto al eje óptico, distancia) en
    puntos (x, y) del laboratorio.

    pos_xy        : posición del dron
    rumbo_camara  : hacia dónde apunta la cámara, en el marco del mundo
    """
    ang = rumbo_camara + np.asarray(rumbos, float)
    d = np.asarray(distancias, float)
    return np.column_stack([pos_xy[0] + d * np.cos(ang),
                            pos_xy[1] + d * np.sin(ang)])


# ══════════════════════════════════════════════════════════════
# MAPA DE OBSTÁCULOS
# ══════════════════════════════════════════════════════════════

class MapaObstaculos:
    """
    Acumula detecciones en el mundo con olvido exponencial.

    El olvido importa: MiDaS se equivoca, y un falso positivo que se quede
    para siempre desvía la ruta el resto del vuelo. Con vida media corta,
    un obstáculo real —que se ve muchas veces seguidas— se refuerza, y uno
    falso se apaga solo.
    """

    def __init__(self, vida_media=2.0, radio_fusion=0.15, max_puntos=400):
        self.vida = float(vida_media)
        self.radio = float(radio_fusion)
        self.max_puntos = int(max_puntos)
        self.pts = np.zeros((0, 2))
        self.peso = np.zeros(0)
        self.t = None

    def decaer(self, t_ahora):
        """
        Aplica el olvido y tira lo que ya no pesa. HAY QUE LLAMARLO CADA
        CICLO, haya detecciones nuevas o no.

        ═══ ESTE ERA EL FALLO MÁS GRAVE DE LA EVASIÓN ═══
        El decaimiento vivía DENTRO de `actualizar()`, y a `actualizar()`
        solo se le llamaba cuando llegaban detecciones nuevas que habían
        sobrevivido a todos los filtros. O sea que un obstáculo confirmado
        se quedaba con su peso intacto PARA SIEMPRE si:

          · dejaban de llegar detecciones,
          · la cámara dejaba de entregar mapas,
          · el objeto salía del encuadre,
          · MiDaS fallaba en silencio,
          · o cualquier detección nueva se caía por el filtro de zona.

        El docstring de esta clase promete «olvido exponencial» y «uno
        falso se apaga solo». No era verdad: un fantasma alcanzaba peso
        alto y se congelaba ahí el resto del vuelo. Medido el 16-sep: la
        zanahoria al tope de 40 cm durante 1215 de 1620 ciclos, el 75 % del
        vuelo, con el obstáculo confirmado a 1.12 m del trípode real.

        Separar el olvido del alta es lo que hace que la promesa se cumpla.
        """
        if self.t is not None:
            dt = max(float(t_ahora) - self.t, 0.0)
            if dt > 0:
                self.peso *= 0.5 ** (dt / self.vida)
        self.t = float(t_ahora)
        if len(self.pts):
            vivos = self.peso > 0.05
            self.pts, self.peso = self.pts[vivos], self.peso[vivos]

    def actualizar(self, nuevos, t_ahora, peso=1.0):
        self.decaer(t_ahora)

        for p in np.asarray(nuevos, float).reshape(-1, 2):
            if len(self.pts):
                d = np.linalg.norm(self.pts - p, axis=1)
                j = int(np.argmin(d))
                if d[j] < self.radio:               # fusionar con el vecino
                    w = self.peso[j] + peso
                    self.pts[j] = (self.pts[j] * self.peso[j] + p * peso) / w
                    self.peso[j] = w
                    continue
            self.pts = np.vstack([self.pts, p])
            self.peso = np.append(self.peso, peso)

        vivos = self.peso > 0.05
        self.pts, self.peso = self.pts[vivos], self.peso[vivos]
        if len(self.pts) > self.max_puntos:
            orden = np.argsort(-self.peso)[:self.max_puntos]
            self.pts, self.peso = self.pts[orden], self.peso[orden]

    def confirmados(self, peso_min=2.0):
        """Solo los vistos varias veces: filtra el ruido de un frame."""
        m = self.peso >= peso_min
        return self.pts[m], self.peso[m]


# ══════════════════════════════════════════════════════════════
# DESVÍO DE LA ZANAHORIA
# ══════════════════════════════════════════════════════════════

def desvio_lateral(p_carrot, tangente, obstaculos, pesos,
                   radio_influencia=0.60, radio_seguro=0.30,
                   desvio_max=0.60, holgura=0.50, lado=0, cola=0.35,
                   margen_lado=0.05):
    """
    Cuánto hay que apartar la zanahoria, PERPENDICULARMENTE a la ruta,
    para pasar a `holgura` metros del obstáculo.

    Por qué solo perpendicular y no un vector libre: desplazar a lo largo
    de la ruta cambiaría la velocidad y rompería el anclaje que impide que
    la zanahoria se desboque. Apartándola de lado, la ruta se sigue
    recorriendo igual y el desvío vuelve a cero solo en cuanto el obstáculo
    queda atrás.

    ══════════════════════════════════════════════════════════════
    POR QUÉ SE CAMBIÓ ESTO (17-sep-2026): el dron rozaba el trípode
    ══════════════════════════════════════════════════════════════
    La versión anterior era un campo de potencial normalizado:

        fuerza = (R_inf − dist) / (R_inf − R_seg)          # dist TOTAL
        desvío = Σ(signo·fuerza·peso)/Σ(peso) · desvío_max

    Tenía dos fallos, y los dos EMPEQUEÑECÍAN el desvío justo cuando
    hacía falta que fuera grande.

    1) REALIMENTACIÓN NEGATIVA. `dist` es la distancia TOTAL de la
       zanahoria al obstáculo. Al apartarse, esa distancia crece, la
       fuerza baja y el desvío se encoge. El sistema no se aparta lo que
       haga falta: se queda en el punto de equilibrio del campo.

       Con R_inf=0.60, R_seg=0.30 y desvío_max=0.40, para un obstáculo
       justo encima de la ruta el equilibrio sale en d = 0.343 m:

           d = clip((0.60 − d)/0.30, 0, 1) · 0.40   →   d = 0.343

       O sea que pasaba a 34 cm del punto detectado por muy alto que se
       pusiera `desvio_max`. Las patas de un trípode se abren ~30 cm
       desde ahí y el dron mide ~10 cm de radio con hélices: los 34 cm
       no daban ni para tocarse de lado. Por eso chocaba AUNQUE lo
       estuviera viendo y esquivando.

    2) LA MEDIA DILUYE. Un trípode no da un punto, da un racimo. Con
       `Σ(fuerza·peso)/Σ(peso)`, los puntos del racimo que todavía
       quedan lejos (fuerza baja) tiran la media abajo y le restan
       autoridad a los que están cerca. Un obstáculo bien visto acababa
       empujando MENOS que uno visto de refilón.

    Ahora el cálculo es geométrico y explícito. Para cada obstáculo se
    mide su posición LATERAL `l` respecto a la ruta, y el desvío que
    haría falta para dejarlo a `holgura`:

        por la izquierda:  l + holgura
        por la derecha:    l − holgura

    Se elige el lado que menos desvío exija —en conjunto, no obstáculo a
    obstáculo— y se toma el MÁXIMO de lo que pide cada uno, no la media:
    despejar a uno no vale si deja a otro dentro.

    La rampa de entrada va con la distancia A LO LARGO de la ruta, que
    NO cambia al apartarse. Ahí muere la realimentación negativa: el
    desvío ya no se desinfla según se ejecuta.

    `holgura` se mide del punto detectado al CENTRO del dron, así que
    tiene que incluir el radio del obstáculo, el del dron y un margen.

    Devuelve (desvio_metros, distancia_al_mas_cercano, no_cabe).
      desvio > 0 → hacia la izquierda del avance.
      no_cabe    → el hueco es más estrecho que la holgura pedida: NO hay
                   desvío que sirva, y el que llama tiene que PARAR en vez
                   de colarse. Sale, por ejemplo, con dos trípodes a un
                   lado y otro de la ruta: pasar por en medio es chocar
                   con uno de los dos, y apartarse más de `desvio_max`
                   tampoco es una opción.
    """
    if len(obstaculos) == 0:
        return 0.0, np.inf, False

    T = np.asarray(tangente, float)[:2]
    n = np.linalg.norm(T)
    if n < 1e-9:
        return 0.0, np.inf, False
    T = T / n
    N = np.array([-T[1], T[0]])           # normal izquierda

    rel = np.asarray(obstaculos, float) - np.asarray(p_carrot, float)[:2]
    dist = np.linalg.norm(rel, axis=1)
    d_min = float(dist.min())

    a = rel @ T                # distancia A LO LARGO de la ruta
    l = rel @ N                # posición lateral (>0 = a la izquierda)

    # Cuentan los que están por delante y dentro del alcance longitudinal.
    # El alcance se mide sobre `a`, no sobre `dist`: ver el punto 1 arriba.
    #
    # `cola` es cuánto se sigue empujando DESPUÉS de que el obstáculo quede
    # por detrás de la zanahoria. No es un detalle: la zanahoria va por
    # delante del dron, así que cuando ella ya ha pasado el trípode, el DRON
    # todavía está a su altura. Con cola=0.10 el desvío se soltaba justo ahí
    # y el dron volvía a la ruta rozando el trípode en vez de rodearlo.
    activo = ((a > -cola) & (a < radio_influencia)
              & (np.abs(l) < desvio_max + holgura))
    if not np.any(activo):
        return 0.0, d_min, False

    a, l = a[activo], l[activo]
    w = np.asarray(pesos, float)[activo]

    # Rampa: 0 al entrar en el radio de influencia, 1 al llegar al radio
    # seguro. Depende solo de `a`, así que apartarse no la reduce. Por
    # detrás (a<0) se queda en 1: el desvío se mantiene mientras el dron
    # termina de pasar, y quien lo devuelve a la ruta es el limitador de
    # pendiente del que llama, suavemente.
    rampa = np.clip((radio_influencia - a)
                    / max(radio_influencia - radio_seguro, 1e-6), 0.0, 1.0)

    # Desvío que pide cada obstáculo por cada lado. El máximo manda: si un
    # obstáculo necesita 50 cm y otro 20, hay que dar 50.
    pide_izq = float(np.max(l + holgura))      # apartarse a la IZQUIERDA
    pide_der = float(np.min(l - holgura))      # apartarse a la DERECHA

    # ── ELECCIÓN DE LADO, Y POR QUÉ SE PUEDE FORZAR ──
    # `lado` viene del que llama: +1 izquierda, −1 derecha, 0 = elige tú.
    #
    # Dejarlo elegir en cada ciclo era el fallo de "se va de izquierda a
    # derecha y nunca lo rodea". El racimo de puntos del trípode cambia un
    # poco en cada cuadro —se ve desde otro ángulo, entran y salen puntos—,
    # y con él cambian `max(l)` y `min(l)`. En cuanto |pide_izq| y
    # |pide_der| se cruzan, el lado elegido salta, el desvío se va al otro
    # sentido, y el dron acaba haciendo eses delante del obstáculo sin
    # pasarlo nunca. Por eso ahora el que llama se COMPROMETE con un lado y
    # lo mantiene hasta haber pasado.
    if lado > 0:
        bruto = pide_izq
    elif lado < 0:
        bruto = pide_der
    elif abs(pide_izq) > desvio_max and abs(pide_der) <= desvio_max:
        bruto = pide_der             # la preferencia nunca vuelve inviable un lado
    elif abs(pide_izq) <= abs(pide_der) + margen_lado:
        bruto = pide_izq
    else:
        bruto = pide_der
    # Dentro del margen se prefiere izquierda de forma determinista.
    # La derecha solo gana con una ventaja geométrica clara.

    # ¿CABE? Si el desvío que hace falta no entra en el tope, no hay
    # maniobra buena: colarse por un hueco más estrecho que la holgura es
    # chocar, solo que más despacio.
    #
    # Se exige ADEMÁS que el obstáculo esté ya encima (dentro del radio
    # seguro). Sin esa condición, un trípode detectado a 60 cm que pidiera
    # un dedo más que `desvio_max` declaraba "no cabe" desde lejos, el lazo
    # lo trataba como parada, la zanahoria se congelaba y el dron se
    # quedaba ahí moviéndose de lado sin avanzar. Que no quepa por el lado
    # que estoy mirando no es razón para plantarse a medio metro; lo es
    # cuando ya no queda sitio.
    no_cabe = (abs(bruto) > desvio_max + 1e-9
               and float(np.min(a)) < radio_seguro)

    # La rampa se aplica al final, y con la del obstáculo que MÁS empuja.
    desvio = bruto * float(np.max(rampa))
    return float(np.clip(desvio, -desvio_max, desvio_max)), d_min, no_cabe


def salida_lateral_segura(pos, objetivo, tangente, obstaculos, radio=0.22):
    """Recorta el avance frontal; permite salir solo si no se acerca a nada.

    Si ya está dentro del corredor de parada, exige separarse de cada punto
    cercano. Para el resto comprueba el segmento entero, no solo su extremo.
    Dentro del radio, un objeto casi centrado puede requerir un retroceso
    mínimo además de la salida lateral. Se calcula por producto escalar,
    sin cambiar de lado ni tolerar una reducción de distancia.
    Devuelve None cuando es necesario mantener la posición.
    """
    pos = np.asarray(pos, float)[:2]
    T = np.asarray(tangente, float)[:2]
    T = T / max(np.linalg.norm(T), 1e-12)
    N = np.array([-T[1], T[0]])
    delta = N * np.dot(np.asarray(objetivo)[:2] - pos, N)
    if np.linalg.norm(delta) < 1e-9:
        return None
    rel = np.asarray(obstaculos, float).reshape(-1, 2) - pos
    if len(rel):
        dist = np.linalg.norm(rel, axis=1)
        cerca = dist < radio
        acerca = (rel @ delta) > 1e-9
        longitudinal = rel @ T
        if np.any(cerca & acerca):
            if np.any(cerca & acerca & (longitudinal <= 1e-9)):
                return None
            seleccion = cerca & acerca
            retroceso = float(np.max((rel[seleccion] @ delta)/longitudinal[seleccion]))
            # Mantener la salida dentro de la dirección lateral observada:
            # como máximo atan(1/4) = 14 grados de componente hacia atrás.
            if retroceso > .25*np.linalg.norm(delta):
                return None
            delta = delta - retroceso*T
        q = np.clip((rel @ delta) / np.dot(delta, delta), 0, 1)
        minimo = np.linalg.norm(rel - q[:, None]*delta, axis=1)
        if np.any((dist < radio) & ((rel @ delta) > 1e-9)):
            return None
        if np.any((dist >= radio) & (minimo < radio)):
            return None
    return pos + delta


def hay_que_parar(obstaculos, p_dron, tangente, radio_parada=0.22,
                  medio_ancho=0.18):
    """
    ¿Hay algo tan cerca y tan delante que lo prudente es quedarse quieto?

    Es la última red: si el desvío no ha bastado, mejor flotar que seguir.
    Solo mira un corredor estrecho al frente, para no frenar por cosas que
    están al lado y no molestan.
    """
    if len(obstaculos) == 0:
        return False, np.inf
    T = np.asarray(tangente, float)[:2]
    n = np.linalg.norm(T)
    if n < 1e-9:
        return False, np.inf
    T = T / n
    N = np.array([-T[1], T[0]])
    rel = np.asarray(obstaculos, float) - np.asarray(p_dron, float)[:2]
    delante = rel @ T
    lado = np.abs(rel @ N)
    en_corredor = (delante > 0) & (delante < radio_parada) & (lado < medio_ancho)
    if not np.any(en_corredor):
        return False, float(np.min(delante[delante > 0])
                            if np.any(delante > 0) else np.inf)
    return True, float(np.min(delante[en_corredor]))
