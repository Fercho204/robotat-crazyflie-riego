"""Máquina de evasión pura, sin cámara, radio, reloj global ni conexiones."""
import numpy as np
from evasion import desvio_lateral, salida_lateral_segura


def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def segmento_libre(p, q, obstaculos, holgura):
    obs = np.asarray(obstaculos, float).reshape(-1, 2)
    if not len(obs): return True
    d = np.asarray(q)-p
    u = np.clip((obs-p) @ d / max(float(d @ d), 1e-12), 0, 1)
    return bool(np.all(np.linalg.norm(obs-p-u[:, None]*d, axis=1) >= holgura-1e-6))


class TramoSuave:
    """Quíntica reposo-reposo con cotas analíticas de rapidez/aceleración."""
    def __init__(self, inicio, fin, t, v_max, a_max, t_min):
        self.inicio, self.fin = np.array(inicio, float), np.array(fin, float)
        d = float(np.linalg.norm(self.fin-self.inicio))
        self.duracion = max(t_min, 1.875*d/v_max, np.sqrt(5.774*d/a_max))
        self.t = t

    def muestra(self, t):
        u = np.clip((t-self.t)/self.duracion, 0, 1)
        f = 10*u**3 - 15*u**4 + 6*u**5
        return self.inicio + f*(self.fin-self.inicio), bool(u >= 1)

    def velocidad(self, t):
        u = np.clip((t-self.t)/self.duracion, 0, 1)
        return (30*u**2-60*u**3+30*u**4)*(self.fin-self.inicio)/self.duracion


class ManiobraEvasion:
    def __init__(self, *, holgura, desvio_max, radio_influencia, radio_seguro,
                 cola, compromiso_min, v_max, a_max, t_min, tol_pos, tol_yaw,
                 edad_max=1.0, espera_max=25.0, margen_lado=.05, espera_vista_max=3.0):
        self.holgura, self.desvio_max = holgura, desvio_max
        self.geometria = dict(holgura=holgura, desvio_max=desvio_max,
                             radio_influencia=radio_influencia,
                             radio_seguro=radio_seguro, cola=cola,
                             margen_lado=margen_lado)
        self.compromiso_min = compromiso_min
        self.v_max, self.a_max, self.t_min = v_max, a_max, t_min
        self.tol_pos, self.tol_yaw = tol_pos, tol_yaw
        self.edad_max, self.espera_max = edad_max, espera_max
        self.espera_vista_max = espera_vista_max
        self.estado = 'RUTA'
        self.lado = 0
        self.historial = ['RUTA']
        self.motivo = ''

    @property
    def activa(self): return self.estado != 'RUTA'

    def cambiar(self, estado, t):
        self.estado, self.desde = estado, t
        self.historial.append(estado)

    def iniciar(self, pos, p_carrot, tangente, obstaculos, pesos, t, lado=0):
        if self.activa: return True
        d, _, no_cabe = desvio_lateral(p_carrot, tangente, obstaculos, pesos,
                                      lado=lado, **self.geometria)
        if abs(d) < self.compromiso_min: return False
        self.reintentos = 0
        self.diag = {}
        self.lado = int(lado or np.sign(d))
        self.T = np.asarray(tangente, float)[:2]
        self.T = self.T / np.linalg.norm(self.T)
        self.N = np.array([-self.T[1], self.T[0]])
        self.origen = np.asarray(p_carrot, float)[:2].copy()
        self.fijo = np.asarray(pos, float)[:2].copy()
        self.obst_iniciales = np.asarray(obstaculos, float).reshape(-1, 2).copy()
        self.yaw_ruta = float(np.arctan2(self.T[1], self.T[0]))
        self.yaw_lado = wrap(self.yaw_ruta + self.lado*np.pi/2)
        self.yaw_obj = self.yaw_lado
        self.cambiar('GIRO_LADO', t)
        if no_cabe: self.bloquear('La holgura no cabe por el lado elegido', t)
        return True

    def bloquear(self, motivo, t):
        self.motivo = motivo
        self.tras_freno = 'BLOQUEADA'
        if self.estado in ('AVANCE_LATERAL', 'AVANCE_PARALELO', 'RETORNO'):
            self.iniciar_freno('BLOQUEADA')
        elif self.estado not in ('BLOQUEADA', 'FRENANDO'):
            self.cambiar('BLOQUEADA', t)

    def iniciar_freno(self, destino):
        self.tras_freno = destino
        self.freno_inicio = self.fijo.copy()
        self.freno_vel = self.tramo.velocidad(self.t_mov)
        self.freno_duracion = max(np.linalg.norm(self.freno_vel)/self.a_max, 1e-6)
        self.cambiar('FRENANDO', self.t_mov)

    def frenar(self, t):
        dt = min(max(t-self.desde, 0), self.freno_duracion)
        self.fijo = self.freno_inicio + self.freno_vel*(dt-.5*dt*dt/self.freno_duracion)
        if dt >= self.freno_duracion: self.cambiar(self.tras_freno, t)

    def fresca(self, vista, t):
        return (vista is not None and vista.get('valida', False)
                and self.desde < vista['ts'] <= t
                and t-vista['ts'] <= self.edad_max)

    def mover(self, destino, estado, t):
        self.tramo = TramoSuave(self.fijo, destino, t, self.v_max, self.a_max, self.t_min)
        self.t_mov = t
        self.cambiar(estado, t)

    def paso(self, pos, yaw, t, vista, obstaculos, retorno, yaw_estable=True):
        """Devuelve (posición absoluta, yaw mundo); None al terminar.

        vista={ts: captura monotónica, valida: bool, bloqueada: bool}; una
        vista desconocida NUNCA equivale a libre. El llamador limita el yaw.
        """
        if not self.activa: return None
        pos = np.asarray(pos, float)[:2]
        obs = np.asarray(obstaculos, float).reshape(-1, 2)
        # La desaparición por olvido del mapa no prueba haber pasado el objeto.
        memoria = np.vstack([self.obst_iniciales, obs])
        if self.estado == 'BLOQUEADA': return self.fijo.copy(), self.yaw_obj
        if self.estado == 'FRENANDO':
            self.frenar(t)
            return self.fijo.copy(), self.yaw_obj
        if (self.estado.startswith(('GIRO_', 'VERIFICAR_'))
                or self.estado in ('AVANCE_LATERAL', 'AVANCE_PARALELO', 'RETORNO', 'ESPERA_VISTA')):
            self.diag = {
                'err_yaw_deg': (float(np.degrees(abs(wrap(yaw-self.yaw_obj))))
                                if yaw is not None and np.isfinite(yaw) else None),
                'err_pos_cm': float(np.linalg.norm(pos-self.fijo)*100),
                'yaw_estable': yaw_estable,
                'edad_vista_s': (t-vista['ts'] if vista is not None
                                 and vista.get('ts') is not None else None),
            }
        if yaw is None or not np.isfinite(yaw):
            self.bloquear('Sin orientación válida', t)
            if self.estado == 'FRENANDO': self.frenar(t)
            return self.fijo.copy(), self.yaw_obj
        if self.estado.startswith(('GIRO_', 'VERIFICAR_')) and t-self.desde > self.espera_max:
            edad = self.diag['edad_vista_s']
            vista_txt = f'{edad:.1f} s' if edad is not None else 'None'
            motivo = ('Sin giro/vista verificable dentro del plazo '
                      f"(yaw {self.diag['err_yaw_deg']:.1f} grados, "
                      f"pos {self.diag['err_pos_cm']:.1f} cm, "
                      f'yaw estable {yaw_estable}, vista {vista_txt})')
            if self.reintentos < 1:
                self.reintentos += 1
                self.fijo = pos.copy()
                self.motivo = 'Reintento: ' + motivo
                self.cambiar(self.estado, t)
            else:
                self.bloquear(motivo, t)
            return self.fijo.copy(), self.yaw_obj
        giros = {'GIRO_LADO':'VERIFICAR_LADO', 'GIRO_RUTA':'VERIFICAR_RUTA',
                 'GIRO_RETORNO':'VERIFICAR_RETORNO',
                 'GIRO_FINAL':'FINAL'}
        if self.estado in giros:
            if (yaw_estable and abs(wrap(yaw-self.yaw_obj)) <= self.tol_yaw
                    and np.linalg.norm(pos-self.fijo) <= self.tol_pos):
                siguiente = giros[self.estado]
                if siguiente == 'FINAL':
                    self.cambiar('RUTA', t)
                    self.lado = 0
                    return None
                self.cambiar(siguiente, t)
        elif self.estado.startswith('VERIFICAR_'):
            if not self.fresca(vista, t): return self.fijo.copy(), self.yaw_obj
            bloqueada = bool(vista['bloqueada'])
            if self.estado == 'VERIFICAR_LADO':
                if bloqueada: self.bloquear('Lado elegido ocupado; no se cambia de lado', t)
                else:
                    lateral = self.lado*float((self.fijo-self.origen) @ self.N)
                    necesario = float(np.max(self.lado*((memoria-self.origen) @ self.N))) + self.holgura
                    distancia = min(max(self.holgura, necesario-lateral), self.desvio_max-lateral)
                    destino = self.fijo + self.lado*distancia*self.N
                    escape = salida_lateral_segura(self.fijo, destino, self.T, memoria,
                                                    radio=self.holgura)
                    if distancia <= self.tol_pos or escape is None:
                        self.bloquear('Salida lateral sin espacio verificado', t)
                    else: self.mover(escape, 'AVANCE_LATERAL', t)
            elif self.estado == 'VERIFICAR_RUTA':
                # Se mira el corredor completo que se pretende recorrer.
                avance = max(self.holgura, float(np.max((self.obst_iniciales-self.fijo) @ self.T)) + self.holgura)
                destino = self.fijo + avance*self.T
                libre = not bloqueada and segmento_libre(self.fijo, destino, memoria, self.holgura)
                if libre:
                    self.mover(destino, 'AVANCE_PARALELO', t)
                else:
                    self.yaw_obj = self.yaw_lado
                    self.cambiar('GIRO_LADO', t)
            elif self.estado == 'VERIFICAR_RETORNO':
                if bloqueada or not segmento_libre(self.fijo, self.retorno, memoria, self.holgura):
                    self.bloquear('Retorno ocupado', t)
                else: self.mover(self.retorno, 'RETORNO', t)
        elif self.estado == 'ESPERA_VISTA':
            espera = t-self.desde
            if espera > self.espera_vista_max:
                self.bloquear(f'Visión ausente durante el avance ({espera:.1f} s)', t)
            elif self.fresca(vista, t):
                if vista['bloqueada']:
                    self.bloquear('Obstáculo nuevo durante el avance', t)
                else: self.mover(self.destino_pendiente, self.estado_pendiente, t)
        elif self.estado in ('AVANCE_LATERAL', 'AVANCE_PARALELO', 'RETORNO'):
            # Una pérdida de visión o un obstáculo nuevo detiene el perfil;
            # se conserva la última referencia, nunca se salta al final.
            if vista is None or not vista.get('valida', False) or t-vista['ts'] > self.edad_max:
                self.destino_pendiente = self.tramo.fin.copy()
                self.estado_pendiente = self.estado
                self.iniciar_freno('ESPERA_VISTA')
            elif vista['bloqueada']:
                self.bloquear('Obstáculo nuevo durante el avance', t)
            else:
                self.fijo, terminado = self.tramo.muestra(t)
                self.t_mov = t
                if terminado and np.linalg.norm(pos-self.tramo.fin) <= self.tol_pos:
                    anterior = self.estado
                    if anterior == 'RETORNO':
                        self.yaw_obj = self.yaw_ruta
                        self.cambiar('GIRO_FINAL', t)
                    elif anterior == 'AVANCE_LATERAL':
                        self.yaw_obj = self.yaw_ruta
                        self.cambiar('GIRO_RUTA', t)
                    else:
                        self.retorno = np.asarray(retorno, float)[:2].copy()
                        delta = self.retorno-self.fijo
                        self.yaw_obj = float(np.arctan2(delta[1], delta[0]))
                        self.cambiar('GIRO_RETORNO', t)
                elif t-self.desde > self.tramo.duracion+self.espera_max:
                    self.bloquear('No se alcanzó el destino', t)
        if self.estado == 'FRENANDO': self.frenar(t)
        return self.fijo.copy(), self.yaw_obj
