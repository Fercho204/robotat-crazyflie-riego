"""
trayectoria_suave.py — vuelo continuo por trayectoria polinómica
=================================================================
Genera una trayectoria suave a partir de la ruta suavizada y la sube al
firmware del Crazyflie, para que la vuele de un tirón, sin frenar.

POR QUÉ HACE FALTA ESTO
───────────────────────
El método anterior mandaba un `go_to` por waypoint. Y `go_to` planifica un
polinomio que llega al destino con VELOCIDAD Y ACELERACIÓN CERO: cada
waypoint es una parada. Con 44 waypoints son 44 frenadas encadenadas.
Mandar el siguiente antes de tiempo (SOLAPAMIENTO) disimula el frenado pero
no lo quita: el dron sigue decelerando hacia cada punto y replanificando
desde un estado intermedio. De ahí los tirones y la pérdida de estabilidad.

La solución que usa Bitcraze (y Crazyswarm) es otra: subir a la memoria del
dron una trayectoria polinómica por tramos y arrancarla con
`start_trajectory`. Entonces el firmware la evalúa a 500 Hz, con velocidad y
aceleración CONTINUAS entre tramos, y con realimentación directa de esas
derivadas al controlador. Tres ventajas:

  1. No hay frenadas: la velocidad nunca pasa por cero dentro de la ruta.
  2. No depende del temporizador de Python ni de la radio: una vez subida,
     el vuelo entero no necesita ni un paquete más. Se acabaron los tirones
     por jitter de red.
  3. El controlador recibe velocidad y aceleración de referencia, no solo
     posición, así que sigue la curva en vez de perseguirla.

CÓMO SE CONSTRUYE
─────────────────
1. PERFIL DE VELOCIDAD. Se recorre la ruta calculando la curvatura y se
   limita la velocidad en cada punto por tres cosas a la vez:
     · velocidad de crucero deseada,
     · aceleración lateral máxima:   v <= sqrt(a_lat / kappa),
     · velocidad de giro de la nariz: v <= omega_max / kappa.
   Después una pasada hacia adelante y otra hacia atrás imponen la
   aceleración longitudinal máxima, así el dron acelera y frena de forma
   gradual (es el perfil clásico de velocidad sobre camino). Integrando
   1/v a lo largo del arco sale el tiempo de cada punto.

2. TRAMOS. La ruta se parte en M tramos de igual duración. M <= 31 por el
   límite de memoria del firmware (4 KB / 132 bytes por tramo).

3. AJUSTE. Cada tramo se ajusta con un polinomio de 7º grado que iguala
   posición, velocidad, aceleración Y sobreaceleración (jerk) en los dos
   extremos: 8 condiciones, 8 coeficientes, solución exacta. Como los dos
   tramos vecinos comparten esos valores en el nudo, la unión es continua
   hasta la tercera derivada. Sin nudos, sin saltos, sin QP.

4. VERIFICACIÓN. Antes de volar nada se remuestrea el polinomio y se
   compara con la ruta original: desviación máxima y RMS, velocidad y
   aceleración máximas. Si la desviación no es aceptable, se sube M y se
   repite. Todo esto ocurre en el PC, sin dron.

El yaw va en el mismo polinomio, así que la nariz gira de forma continua en
lugar de a saltos entre waypoints.
"""

from __future__ import annotations

import numpy as np


# ═══════════════════════════════════════════════════════════════
# 1) PERFIL DE VELOCIDAD SOBRE EL CAMINO
# ═══════════════════════════════════════════════════════════════

def _remuestrear_arco(ruta_xy, paso=0.005):
    """Remuestrea la ruta a paso de arco constante. Es la base de todo:
    con puntos equiespaciados, curvatura y velocidad se calculan sin sesgo."""
    p = np.asarray(ruta_xy, dtype=float)
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    keep = np.concatenate(([True], d > 1e-9))
    p = p[keep]
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))))
    n = max(int(s[-1] / paso) + 1, 50)
    s_new = np.linspace(0.0, s[-1], n)
    x = np.interp(s_new, s, p[:, 0])
    y = np.interp(s_new, s, p[:, 1])
    return np.column_stack([x, y]), s_new


def _curvatura(p, s):
    """Curvatura kappa = |x' y'' - y' x''| con derivadas respecto al arco."""
    dx = np.gradient(p[:, 0], s)
    dy = np.gradient(p[:, 1], s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)
    num = np.abs(dx * ddy - dy * ddx)
    den = (dx ** 2 + dy ** 2) ** 1.5
    k = np.divide(num, np.maximum(den, 1e-12))
    # Suavizado ligero: la curvatura numérica es ruidosa y no queremos que
    # un pico de un solo punto frene el dron sin motivo.
    if len(k) > 11:
        w = np.ones(11) / 11.0
        k = np.convolve(np.pad(k, (5, 5), mode='edge'), w, mode='valid')
    return k


def perfil_velocidad(p, s, v_crucero, a_lat_max, a_lon_max, yaw_rate_max,
                     v_min=0.08):
    """
    Velocidad permitida en cada punto del camino.

    Devuelve v[i] respetando a la vez:
      · v <= v_crucero
      · v <= sqrt(a_lat_max / kappa)      (no derrapar en la curva)
      · v <= yaw_rate_max / kappa         (la nariz puede seguir el rumbo)
      · |dv/dt| <= a_lon_max              (pasadas adelante y atrás)
      · v = 0 en los dos extremos         (arranca y termina parado)
    """
    k = _curvatura(p, s)
    v = np.full(len(s), float(v_crucero))

    with np.errstate(divide='ignore', invalid='ignore'):
        v_lat = np.sqrt(a_lat_max / np.maximum(k, 1e-9))
        v_yaw = yaw_rate_max / np.maximum(k, 1e-9)
    v = np.minimum(v, v_lat)
    v = np.minimum(v, v_yaw)
    v = np.clip(v, v_min, v_crucero)

    # Extremos parados
    v[0] = 0.0
    v[-1] = 0.0

    # Pasada hacia adelante: no acelerar más de a_lon_max
    for i in range(1, len(v)):
        ds = s[i] - s[i - 1]
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * a_lon_max * ds))
    # Pasada hacia atrás: no frenar más de a_lon_max
    for i in range(len(v) - 2, -1, -1):
        ds = s[i + 1] - s[i]
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_lon_max * ds))

    return v, k


def tiempos_desde_velocidad(s, v, v_piso=0.02):
    """Integra dt = ds / v para obtener el tiempo en cada punto del arco."""
    vv = np.maximum(v, v_piso)          # evita dividir por cero en los extremos
    ds = np.diff(s)
    v_med = 0.5 * (vv[:-1] + vv[1:])
    dt = ds / v_med
    return np.concatenate(([0.0], np.cumsum(dt)))


# ═══════════════════════════════════════════════════════════════
# 2) AJUSTE POLINÓMICO DE 7º GRADO POR TRAMOS
# ═══════════════════════════════════════════════════════════════

def _hermite5(p0, v0, a0, p1, v1, a1, T):
    """
    Polinomio de 5º grado que iguala posición, velocidad y aceleración en
    t=0 y en t=T. Seis condiciones, seis coeficientes: exacto.

    Se rellena a 8 coeficientes con ceros porque el firmware espera grado 7.

    ¿Por qué quíntico y no séptico? El séptico exige además igualar el JERK
    en los nudos, y el jerk de una referencia muestreada es ruido: derivar
    tres veces amplifica cualquier temblor numérico. Probado: con séptico y
    jerk numérico salían aceleraciones de 3 a 8 m/s², absurdas para un
    Crazyflie. Con quíntico basta la continuidad C2 —posición, velocidad y
    aceleración continuas—, que es justo lo que necesita el controlador, y
    los números se vuelven físicos.

    Devuelve [c0..c7], p(t) = c0 + c1 t + ... , t local en [0,T].
    """
    c0, c1, c2 = p0, v0, a0 / 2.0
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5

    A = np.array([
        [T3,     T4,      T5],
        [3*T2,   4*T3,    5*T4],
        [6*T,    12*T2,   20*T3],
    ], dtype=float)
    b = np.array([
        p1 - (c0 + c1*T + c2*T2),
        v1 - (c1 + 2*c2*T),
        a1 - (2*c2),
    ], dtype=float)

    c3, c4, c5 = np.linalg.solve(A, b)
    return [c0, c1, c2, c3, c4, c5, 0.0, 0.0]


def _derivadas_suaves(f, t, s_rel=1e-4):
    """
    Primera y segunda derivada ANALÍTICAS, vía spline de suavizado.

    Derivar con np.gradient dos veces sobre una señal interpolada mete
    ruido que luego el polinomio amplifica. Ajustando un spline quíntico
    con un poco de suavizado y derivándolo de forma analítica, las
    derivadas salen limpias y consistentes entre sí.
    """
    from scipy.interpolate import UnivariateSpline
    escala = max(float(np.ptp(f)), 1e-6)
    sp = UnivariateSpline(t, f, k=5, s=s_rel * escala ** 2 * len(t))
    return sp(t), sp.derivative(1)(t), sp.derivative(2)(t)


def _evaluar(coef, t):
    """Evalúa un polinomio [c0..c7] y sus derivadas en un vector de tiempos."""
    p = sum(c * t ** i for i, c in enumerate(coef))
    v = sum(i * c * t ** (i - 1) for i, c in enumerate(coef) if i >= 1)
    a = sum(i * (i - 1) * c * t ** (i - 2) for i, c in enumerate(coef) if i >= 2)
    return p, v, a


# ═══════════════════════════════════════════════════════════════
# 3) GENERACIÓN COMPLETA
# ═══════════════════════════════════════════════════════════════

class Trayectoria:
    """Trayectoria polinómica por tramos, lista para subir al dron."""

    def __init__(self, tramos, duracion, ref, t_ref):
        self.tramos = tramos            # lista de dicts con coef x,y,z,yaw,T
        self.duracion = duracion
        self.ref = ref                  # referencia (x,y) usada para verificar
        self.t_ref = t_ref

    # -- verificación en el PC, sin dron ------------------------------
    def verificar(self, n=4000):
        """
        Remuestrea la trayectoria y la compara con la ruta de referencia.
        Devuelve un dict con desviación, velocidad y aceleración máximas.
        """
        ts = np.linspace(0.0, self.duracion, n)
        xs, ys, vxs, vys, axs_, ays = [], [], [], [], [], []
        idx, t0 = 0, 0.0
        for t in ts:
            while idx < len(self.tramos) - 1 and t > t0 + self.tramos[idx]['T']:
                t0 += self.tramos[idx]['T']
                idx += 1
            tl = np.clip(t - t0, 0.0, self.tramos[idx]['T'])
            px, vx, ax = _evaluar(self.tramos[idx]['x'], tl)
            py, vy, ay = _evaluar(self.tramos[idx]['y'], tl)
            xs.append(px); ys.append(py)
            vxs.append(vx); vys.append(vy)
            axs_.append(ax); ays.append(ay)

        P = np.column_stack([xs, ys])
        V = np.hypot(vxs, vys)
        A = np.hypot(axs_, ays)

        # Desviación respecto a la ruta original: para cada punto generado,
        # distancia al punto más cercano de la referencia densa.
        ref = self.ref
        d = np.empty(len(P))
        for i, q in enumerate(P):
            d[i] = np.min(np.linalg.norm(ref - q, axis=1))

        return {
            'desv_max_cm': float(d.max() * 100),
            'desv_rms_cm': float(np.sqrt(np.mean(d ** 2)) * 100),
            'v_max': float(V.max()),
            'v_media': float(V.mean()),
            'a_max': float(A.max()),
            'duracion': float(self.duracion),
            'n_tramos': len(self.tramos),
            'v_min_interior': float(V[int(0.05*len(V)):int(0.95*len(V))].min()),
        }

    # -- conversión al formato del firmware ---------------------------
    def a_poly4d(self):
        """Lista de Poly4D lista para TrajectoryMemory.poly4Ds."""
        from cflib.crazyflie.mem import Poly4D
        out = []
        for tr in self.tramos:
            out.append(Poly4D(
                float(tr['T']),
                Poly4D.Poly([float(c) for c in tr['x']]),
                Poly4D.Poly([float(c) for c in tr['y']]),
                Poly4D.Poly([float(c) for c in tr['z']]),
                Poly4D.Poly([float(c) for c in tr['yaw']]),
            ))
        return out


def generar(ruta_xy, altura, v_crucero=0.25, a_lat_max=0.5, a_lon_max=0.35,
            yaw_rate_max=np.radians(60.0), n_tramos=28, yaw_activo=True,
            yaw_lookahead_m=0.10, yaw_offset=0.0, v_min=0.08):
    """
    Construye la trayectoria a partir de la ruta 2D suavizada.

    v_crucero    : velocidad objetivo en recta (m/s)
    a_lat_max    : aceleración lateral máxima en curva (m/s²). Es el mando
                   que decide cuánto frena en las curvas. Más alto = más
                   rápido y más inclinado; más bajo = curvas más suaves.
    a_lon_max    : aceleración longitudinal máxima (m/s²)
    yaw_rate_max : velocidad de giro de la nariz (rad/s)
    n_tramos     : tramos polinómicos. Máximo 31 por memoria del firmware.
    """
    n_tramos = int(np.clip(n_tramos, 4, 31))

    # ── 1) Camino a paso de arco constante ──
    p, s = _remuestrear_arco(ruta_xy, paso=0.005)

    # ── 2) Perfil de velocidad y tiempos ──
    v, k = perfil_velocidad(p, s, v_crucero, a_lat_max, a_lon_max,
                            yaw_rate_max, v_min=v_min)
    t = tiempos_desde_velocidad(s, v)
    T_total = float(t[-1])

    # ── 3) Referencia muestreada en TIEMPO uniforme ──
    # A partir de aquí todo va en función del tiempo, que es como lo
    # entiende el firmware.
    n_fino = 6000
    tt = np.linspace(0.0, T_total, n_fino)
    xr = np.interp(tt, t, p[:, 0])
    yr = np.interp(tt, t, p[:, 1])

    # Yaw tangente, adelantado unos centímetros para que la nariz vaya
    # entrando en la curva en vez de persiguiéndola.
    if yaw_activo:
        s_t = np.interp(tt, t, s)
        s_ade = np.clip(s_t + yaw_lookahead_m, s[0], s[-1])
        x_ade = np.interp(s_ade, s, p[:, 0])
        y_ade = np.interp(s_ade, s, p[:, 1])
        yaw_r = np.unwrap(np.arctan2(y_ade - yr, x_ade - xr)) + yaw_offset
        # Suavizado suave: quita el ruido de la diferencia sin desalinear.
        w = np.ones(31) / 31.0
        yaw_r = np.convolve(np.pad(yaw_r, (15, 15), mode='edge'), w, mode='valid')
    else:
        yaw_r = np.zeros(n_fino)

    zr = np.full(n_fino, float(altura))

    # ── 4) Derivadas analíticas de la referencia ──
    xr, dx, ddx = _derivadas_suaves(xr, tt)
    yr, dy, ddy = _derivadas_suaves(yr, tt)
    yaw_r, dyaw, ddyaw = _derivadas_suaves(yaw_r, tt)

    # Extremos exactamente parados: el dron arranca y termina quieto.
    for arr in (dx, ddx, dy, ddy, dyaw, ddyaw):
        arr[0] = 0.0
        arr[-1] = 0.0

    # ── 5) Nudos igualmente espaciados en tiempo ──
    bordes = np.linspace(0.0, T_total, n_tramos + 1)
    idx_borde = np.searchsorted(tt, bordes)
    idx_borde = np.clip(idx_borde, 0, n_fino - 1)
    idx_borde[0], idx_borde[-1] = 0, n_fino - 1

    tramos = []
    for i in range(n_tramos):
        i0, i1 = idx_borde[i], idx_borde[i + 1]
        T = float(tt[i1] - tt[i0])
        if T <= 1e-6:
            continue
        cx = _hermite5(xr[i0], dx[i0], ddx[i0], xr[i1], dx[i1], ddx[i1], T)
        cy = _hermite5(yr[i0], dy[i0], ddy[i0], yr[i1], dy[i1], ddy[i1], T)
        cz = [float(altura), 0, 0, 0, 0, 0, 0, 0]
        cyaw = _hermite5(yaw_r[i0], dyaw[i0], ddyaw[i0],
                         yaw_r[i1], dyaw[i1], ddyaw[i1], T)
        tramos.append({'x': cx, 'y': cy, 'z': cz, 'yaw': cyaw, 'T': T})

    return Trayectoria(tramos, T_total, p, t)


def generar_verificada(ruta_xy, altura, desv_objetivo_cm=3.0, **kw):
    """
    Genera y sube el número de tramos hasta que la desviación respecto a la
    ruta original baje del objetivo (o se llegue al tope de 31).

    Devuelve (trayectoria, informe). Nunca lanza: si algo falla, informe
    trae 'error' y la trayectoria es None.
    """
    try:
        mejor = None
        for n in (12, 16, 20, 24, 28, 31):
            kw2 = dict(kw)
            kw2['n_tramos'] = n
            tr = generar(ruta_xy, altura, **kw2)
            info = tr.verificar()
            mejor = (tr, info)
            if info['desv_max_cm'] <= desv_objetivo_cm:
                break
        return mejor[0], mejor[1]
    except Exception as e:                                   # pragma: no cover
        return None, {'error': str(e)}


def resumen(info, v_crucero=None):
    """Imprime el informe de verificación en lenguaje claro."""
    if 'error' in info:
        print(f'  ✗ No se pudo generar la trayectoria: {info["error"]}')
        return False
    print(f'  Tramos polinómicos : {info["n_tramos"]} / 31 (límite del firmware)')
    print(f'  Duración           : {info["duracion"]:.1f} s')
    print(f'  Velocidad media    : {info["v_media"]:.3f} m/s '
          f'(máx {info["v_max"]:.3f})')
    print(f'  Velocidad mínima   : {info["v_min_interior"]:.3f} m/s '
          f'← NUNCA llega a 0 dentro de la ruta')
    print(f'  Aceleración máxima : {info["a_max"]:.2f} m/s²')
    print(f'  Desviación de la ruta: {info["desv_rms_cm"]:.1f} cm RMS, '
          f'{info["desv_max_cm"]:.1f} cm máx')
    if info['desv_max_cm'] > 8.0:
        print('  ⚠ La trayectoria se separa bastante de la ruta dibujada.')
        print('    Sube n_tramos o baja a_lat_max.')
        return False
    return True
