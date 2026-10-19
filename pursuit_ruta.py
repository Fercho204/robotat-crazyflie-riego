"""
pursuit_ruta.py — Seguimiento de ruta por Pure Pursuit con realimentación
de posición EN EL PC contra el MoCap.

ORIGEN
Portado de `nuevo metodo/Code_V1_backup/{pure_pursuit,trajectory}.py`, que
tiene 15 vuelos medidos detrás. Se reúne aquí en un solo archivo para que
Prueba_ruta_vision.py no dependa de esa carpeta.

POR QUÉ ESTE MÉTODO Y NO go_to NI start_trajectory
Es la diferencia estructural, y explica un factor 50 en el error medido:

  go_to / start_trajectory  → el lazo de posición se cierra DENTRO del
      dron, contra su EKF. Medido en este montaje: 13.9 cm RMS de error de
      estimación, con picos de 60 cm. El dron va exactamente a donde cree
      que está el punto, y cree mal.

  Pure Pursuit              → el lazo de posición se cierra EN EL PC,
      contra la posición del MoCap directamente. El error del EKF no entra
      en el lazo de posición; el EKF solo lo usa el controlador interno de
      velocidad/actitud, que es mucho menos sensible a él.

Resultados medidos con este método en este mismo laboratorio (vuelo real,
métricas guardadas en nuevo metodo/.../vuelos/metricas_vuelo_real.txt):
    error lateral medio 1.2 cm, máximo 3.2 cm, a 0.30 m/s,
    en una ruta con radio mínimo de 0.24 m y con la batería a 3.40 V.

LEY DE CONTROL
    v_xy = rapidez * T(s_proj + L)  +  KP_LAT * e_lat
    v_z  = rapidez * T_z            +  KP_Z   * (z_ref - z)

El primer término es anticipación: la TANGENTE de la curva L metros por
delante. No la cuerda dron→lookahead, que siempre cae por dentro del arco
y es de donde viene el recorte de curvas del Pure Pursuit clásico.
El segundo cierra el lazo sobre la distancia perpendicular a la curva.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════
# GEOMETRÍA DE LA POLILÍNEA
# ═══════════════════════════════════════════════════════════════

def quitar_duplicados(path, tol=1e-6):
    """Puntos consecutivos iguales rompen la tangente (división por cero)."""
    P = np.asarray(path, dtype=float)
    if len(P) < 2:
        return P
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    keep = np.concatenate([[True], d > tol])
    return P[keep]


def remuestrear(path, paso=0.02):
    """Reparte los puntos a espaciado constante sobre el arco."""
    P = quitar_duplicados(path)
    if len(P) < 2:
        return P
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(np.ceil(s[-1] / max(paso, 1e-4))) + 1, 2)
    s_new = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(s_new, s, P[:, k]) for k in range(P.shape[1])],
                    axis=1)


def geometria(path, kappa_suave_m=0.12, kappa_max=25.0):
    """
    s (arco acumulado), T (tangente unitaria 3D), kappa (curvatura XY).

    La curvatura se deriva respecto a s, NO respecto al índice, así que no
    depende del espaciado de los puntos. La segunda derivada numérica es
    ruidosa, por eso se promedia sobre kappa_suave_m metros.
    """
    P = quitar_duplicados(path)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])

    d = np.gradient(P, s, axis=0)
    T = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)

    dx, dy = d[:, 0], d[:, 1]
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)
    den = (dx * dx + dy * dy) ** 1.5
    kappa = np.clip(np.abs(dx * ddy - dy * ddx) / np.maximum(den, 1e-9),
                    0.0, kappa_max)

    ds = float(np.median(np.diff(s))) if len(s) > 2 else 1.0
    w = max(3, int(round(kappa_suave_m / max(ds, 1e-6))) | 1)
    if w < len(kappa):
        pad = w // 2
        kappa = np.convolve(np.pad(kappa, pad, mode='edge'),
                            np.ones(w) / w, mode='valid')
    return P, s, T, kappa


def perfil_velocidad(s, kappa, T, v_max=0.30, v_min=0.10,
                     a_lat=0.35, a_tan=0.40, vz_max=0.30,
                     v_ini=0.12, v_fin=0.10):
    """
    Rapidez a lo largo del arco, con tres límites hechos alcanzables por
    una pasada hacia atrás y otra hacia adelante:

      1. Aceleración lateral    : v <= sqrt(a_lat / kappa)
      2. Velocidad vertical     : v <= vz_max / |dz/ds|
      3. Aceleración tangencial : |dv/dt| <= a_tan en los dos sentidos.

    La pasada HACIA ATRÁS es la que importa: obliga a llegar frenado a la
    curva en vez de frenar dentro de ella.
    """
    s = np.asarray(s, float)
    kappa = np.asarray(kappa, float)
    T = np.asarray(T, float)
    n = len(s)

    v = np.full(n, float(v_max))
    v = np.minimum(v, np.sqrt(a_lat / np.maximum(kappa, 1e-6)))
    v = np.minimum(v, vz_max / np.maximum(np.abs(T[:, 2]), 1e-6))
    v = np.clip(v, v_min, v_max)
    v[0] = min(v[0], float(v_ini))
    v[-1] = min(v[-1], float(v_fin))

    for i in range(n - 2, -1, -1):                       # frenar con antelación
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_tan * (s[i + 1] - s[i])))
    for i in range(1, n):                                # arrancar sin tirón
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * a_tan * (s[i] - s[i - 1])))
    return v


# ═══════════════════════════════════════════════════════════════
# PURE PURSUIT
# ═══════════════════════════════════════════════════════════════

class PurePursuit:
    """
    Base: Coulter, R.C. (1992), Implementation of the Pure Pursuit Path
    Tracking Algorithm, CMU. Con cuatro cambios respecto al clásico:

      1. INDEXADO POR LONGITUD DE ARCO. El punto de anticipación se obtiene
         avanzando L metros sobre la curva desde la proyección del dron, no
         intersectando un círculo de radio L con los segmentos. Cuando el
         error lateral supera L —justo lo que pasa en curva cerrada— el
         círculo no corta nada y el clásico cae en un fallback arbitrario.
         Avanzar L metros sobre la curva siempre tiene solución.
      2. PROYECCIÓN CONTINUA sobre el segmento (parámetro t continuo), no
         el vértice más cercano: error lateral y altura sin escalones.
      3. TANGENTE Y CURVATURA precalculadas respecto al arco y filtradas.
      4. AVANCE ACOTADO del índice: la búsqueda hacia adelante se limita a
         lo que el dron pudo recorrer en un ciclo, para que en una horquilla
         el índice no salte al otro lado de la curva y se coma un tramo.
    """

    def __init__(self, L=0.25, L_min=0.10, L_max=0.45, T_look=0.40,
                 k_curve=0.0, tol_meta=0.10, ventana=0.50):
        self.L = float(L)
        self.L_min = float(L_min)
        self.L_max = float(L_max)
        self.T_look = float(T_look)   # L = T_look * v ≈ retardo del lazo
        self.k_curve = float(k_curve)
        self.tol_meta = float(tol_meta)
        self.ventana = float(ventana)

        self.path = None
        self.s = None
        self.T = None
        self.kappa = None
        self.length = 0.0
        self.reset()

    def set_path(self, path):
        self.path, self.s, self.T, self.kappa = geometria(path)
        self.length = float(self.s[-1])
        self.reset()
        return self

    def reset(self):
        self.index = 0
        self.t_seg = 0.0
        self.s_proj = 0.0
        self.proj = self.path[0].copy() if self.path is not None else None
        self.e_lat = np.zeros(2)
        self.e_lat_signed = 0.0
        self._finished = False

    @property
    def finished(self):
        return self._finished

    def lookahead_adaptativo(self, v, kappa=None):
        """L proporcional a la velocidad: así la anticipación corresponde
        al punto donde el dron estará cuando el comando surta efecto."""
        L = self.T_look * max(float(v), 1e-3)
        if kappa is not None and self.k_curve > 0.0:
            L = L / (1.0 + self.k_curve * float(kappa))
        self.L = float(np.clip(L, self.L_min, self.L_max))
        return self.L

    def step(self, x, y, max_avance=None):
        """Proyecta el dron sobre la ruta. Llamar UNA vez por ciclo."""
        if self.path is None:
            return
        n = len(self.path)
        self.index = min(self.index, n - 2)

        pos = np.array([float(x), float(y)])
        ventana = self.ventana if max_avance is None else float(max_avance)

        i_hi = int(np.searchsorted(self.s, self.s[self.index] + max(ventana, 1e-3),
                                   side='right'))
        i_hi = min(max(i_hi, self.index + 2), n)

        A = self.path[self.index:i_hi - 1, :2]
        B = self.path[self.index + 1:i_hi, :2]
        seg = B - A
        w = pos - A

        den = np.einsum('ij,ij->i', seg, seg)
        num = np.einsum('ij,ij->i', w, seg)
        t = np.clip(np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0), 0.0, 1.0)

        proj_xy = A + t[:, None] * seg
        j = int(np.argmin(np.linalg.norm(pos - proj_xy, axis=1)))

        self.index += j
        self.t_seg = float(t[j])
        i = self.index
        self.s_proj = float(self.s[i] + self.t_seg * (self.s[i + 1] - self.s[i]))
        self.proj = self.path[i] + self.t_seg * (self.path[i + 1] - self.path[i])

        self.e_lat = self.proj[:2] - pos
        Tp = self.estado_en(self.s_proj)[1]
        self.e_lat_signed = float(Tp[0] * self.e_lat[1] - Tp[1] * self.e_lat[0])

        if (self.length - self.s_proj) <= self.tol_meta:
            self._finished = True

    def estado_en(self, s_q):
        """Punto, tangente unitaria y curvatura a un arco dado."""
        s_q = float(np.clip(s_q, 0.0, self.length))
        i = int(np.clip(int(np.searchsorted(self.s, s_q, side='right')) - 1,
                        0, len(self.s) - 2))
        ds = self.s[i + 1] - self.s[i]
        t = 0.0 if ds < 1e-12 else (s_q - self.s[i]) / ds

        p = self.path[i] + t * (self.path[i + 1] - self.path[i])
        T = self.T[i] + t * (self.T[i + 1] - self.T[i])
        nrm = np.linalg.norm(T)
        T = T / nrm if nrm > 1e-9 else self.T[i]
        k = float(self.kappa[i] + t * (self.kappa[i + 1] - self.kappa[i]))
        return p, T, k

    def estado_lookahead(self):
        return self.estado_en(self.s_proj + self.L)

    def curvatura_aqui(self):
        return self.estado_en(self.s_proj)[2]

    def progreso(self):
        if self.path is None or self.length < 1e-9:
            return 0.0
        return 100.0 * self.s_proj / self.length
