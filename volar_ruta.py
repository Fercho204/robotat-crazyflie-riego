"""
volar_ruta.py — VERSIÓN 3 de 3

MÉTODO: "zanahoria móvil". Se calcula en el PC un punto que avanza sobre
la ruta y se le manda al dron como CONSIGNA DE POSICIÓN a 50 Hz
(`send_position_setpoint`). El controlador de posición del dron persigue
esa consigna; el punto nunca se para, así que el dron tampoco frena.

No modifica sistema_vuelo.py: lo importa y le inyecta esta función.
Cámara, paro de emergencia, MoCap, gráficas e informe son idénticos a las
otras dos versiones, así que la comparación es limpia.

    python volar_ruta.py

──────────────────────────────────────────────────────────────────────────
POR QUÉ ESTE TERCER MÉTODO Y NO UNA VARIANTE DE LOS OTROS

Las tres versiones cubren las tres casillas que existen:

  ver.  quién cierra el lazo   contra qué      referencia
  ────  ────────────────────   ─────────────   ──────────────────────────
   1    el PC                  MoCap (real)    velocidad, continua
   2    el dron                su EKF          waypoints sueltos, solapados
   3    el dron                su EKF          POSICIÓN continua, sin parar

La 3 es el híbrido: la inteligencia geométrica (proyección por longitud de
arco, anticipación, perfil de velocidad) se calcula aquí con la posición
REAL del MoCap, pero el seguimiento fino lo hace el controlador de
posición del dron, que corre a 500 Hz y no depende de que el WiFi ni el
GIL de Python vayan finos.

VENTAJA DE SEGURIDAD, y es la razón de peso para probarla:
  · Con consignas de VELOCIDAD (versión 1), si el lazo del PC se atasca
    —y aquí la cámara y MiDaS compiten por CPU— el dron mantiene la última
    velocidad y SE VA VOLANDO en línea recta.
  · Con consignas de POSICIÓN (esta), si el lazo se atasca el dron se
    queda flotando en la última consigna. Se para solo.
  Con una cámara pegada al mismo proceso, esa diferencia importa.

POR QUÉ NO SE DESBOCA (que es lo que mató a start_trajectory)
El polinomio subido al dron decía "en t=32.4 s estás en P" y seguía
avanzando le siguiera el dron o no; si se retrasaba, nadie le esperaba y
el error se acumulaba durante toda la ruta.
Aquí la zanahoria se ancla a la PROYECCIÓN DEL DRON sobre la ruta:

    s_zanahoria = min(s_zanahoria + v·dt,  s_proyección + AVANCE_MAX)

Si el dron se retrasa, su proyección deja de avanzar, y la zanahoria se
queda esperándole a AVANCE_MAX metros. No es una función del tiempo: es
una función de dónde está el dron. No puede escaparse.

Y AVANCE_MAX es también el mando de velocidad efectivo: el controlador de
posición del dron empuja proporcionalmente a la distancia a la consigna,
así que cuanta más zanahoria por delante, más rápido va.
"""

import sys
import time
import numpy as np

import sistema_vuelo as P
from maniobra_evasion import ManiobraEvasion


# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE ESTA VERSIÓN
# coeficientes MiDas:  A = 0.007382 y B = 0.559695 
# Coeficientes Depth Anything V2: A = 0.250945 y B = 0.959041
# ══════════════════════════════════════════════════════════════

P.MODO_EJECUCION = 'pursuit'      # reutiliza el MoCap sin filtrar de la v1

VISION_MODELO = 'hibrido'  # midas / depth_anything_v2 / depth_anything_v2_metrico / hibrido
P.VISION_MODELO = VISION_MODELO
# Lado de entrada de Depth Anything: 336; None usa el del procesador (518).
VISION_DA_LADO = 336
P.VISION_DA_LADO = VISION_DA_LADO
VISION_ESPERA_S = 120.0  # modelo listo y mapas nuevos antes de despegar

# El controlador convierte el error de posición (mundo) en roll/pitch (cuerpo)
# con el yaw del EKF. Sin la orientación del MoCap, ese yaw arranca en 0° tras
# el reset y solo vale si el dron está puesto EXACTAMENTE con ese rumbo: si no,
# corrige torcido y se inclina o vuelca al despegar. Con esto el EKF toma el yaw
# del MoCap y da igual cómo se ponga; antes de despegar se comprueba (abajo).
#
# DESACTIVADO tras un vuelo (23-sep-2026) en que, con esto en True y el yaw ya
# alineado (Δ -2.5°), el dron salió disparado de lado al despegar. Sin
# orientación del MoCap el EKF usa giroscopio+acelerómetro para roll/pitch.
P.USAR_ORIENTACION_MOCAP = False
# El preámbulo (despegue, ir al inicio, mantenerse) es el de
# sistema_vuelo.py en modo 'pursuit', que despega bien en cualquier
# punto: velocidad corregida con el MoCap desde el PC y SIN reiniciar el EKF
# antes de subir (reiniciarlo justo antes lo dejaba sin asentar).
ALINEAR_YAW_ANTES_DE_DESPEGAR = False
# Sin reinicio, el EKF ya está asentado desde la conexión: la compuerta solo
# comprueba 1 s que su posición coincide con la del MoCap.
EKF_ESPERA_MIN_S = 0.0
EKF_ESPERA_MAX_S = 20.0
EKF_TOL_M = 0.03
EKF_ESTABLE_S = 1.0
ALINEA_YAW_TOL_DEG  = 10.0   # ° máx. entre el yaw del EKF y el rumbo real del dron
ALINEA_YAW_ESPERA_S = 8.0    # s que se espera a que el EKF converja
ALINEA_YAW_CONV_S   = 2.0    # s tras reiniciar el EKF, para que converja la posición
DESPEGUE_DESVIO_MAX = 0.50   # m: si en el ascenso se aleja más, se cortan los motores.
                             # (El ascenso solo en velocidad deriva ~25 cm hasta
                             # 0.6 m; la huida real fue de 2.5 m.)
MOCAP_VIVO_ESPERA_S = 5.0    # s que se espera a que Motive rastree el dron en el suelo
DESPEGUE_VZ         = 0.10   # m/s de ascenso, vertical puro como el pursuit

CTRL_HZ      = 50.0    # cadencia con la que se manda la consigna
PASO_RUTA    = 0.02    # m de espaciado al remuestrear la ruta

V_MAX        = 0.10    # m/s en recta
V_MIN        = 0.10    # m/s en la curva más cerrada
A_LAT_MAX    = 0.25    # m/s² lateral (decide la velocidad en curva)
A_TAN_MAX    = 0.40    # m/s² tangencial (frenado anticipado)

# Cuánto puede adelantarse la zanahoria a la proyección del dron.
#
# MEDIDO EN SIMULACIÓN, no supuesto. El controlador de posición del
# Crazyflie manda velocidad proporcional al error (posCtlPid.xKp ≈ 2.0),
# así que la velocidad que sale es simplemente:
#
#       v_real = min( V_MAX , KP_POS × AVANCE_MAX )   con KP_POS ≈ 2.0
#
#   AVANCE_MAX   v_real   e_lat medio
#      0.08 m    0.16 m/s   0.14 cm      ← lo limita la zanahoria
#      0.10 m    0.20 m/s   0.20 cm      ← lo limita la zanahoria
#      0.15 m    0.25 m/s   0.28 cm      ← ya lo limita V_MAX
#      0.20 m    0.25 m/s   0.28 cm      ← idéntico: no cambia nada
#      0.30 m    0.25 m/s   0.28 cm      ← idéntico
#
# O sea que AVANCE_MAX es el mando de velocidad SOLO por debajo de
# V_MAX/KP_POS ≈ 0.125 m. Por encima de eso manda V_MAX y subir la
# zanahoria no hace absolutamente nada. Lo digo porque yo había escrito
# justo lo contrario antes de simularlo.
#
# 0.15 deja un margen sano por encima del punto de saturación (0.125),
# que es lo que se quiere: la zanahoria no debe ser el cuello de botella,
# solo el tope de seguridad.
AVANCE_MAX   = 0.15
AVANCE_MIN   = 0.04    # nunca menos: si llega a cero, el dron se para
RUTA_RAMPA_S = 3.0     # arranque y retorno de maniobra con avance gradual

# ── LA ZANAHORIA EN CURVA: aquí NO sé la respuesta, y hay que medirla ──
# Tu ruta tiene radio mínimo 0.08 m, y ahí compiten dos efectos OPUESTOS:
#
#   · RECORTE POR CUERDA — la línea dron→zanahoria pasa por dentro del
#     arco. La flecha vale ~L²/(8R): con L=15 cm y R=0.08 m son 3.5 cm,
#     que sobre tu error medio de 4.11 cm no es poco. Esto pide una
#     zanahoria MÁS CORTA en curva.
#
#   · RETARDO DEL LAZO — el controlador de posición tarda ~0.35 s. Para
#     recorrer un círculo de R=0.08 a 0.167 m/s hace falta girar a 120°/s,
#     y con ese retardo el dron va ~40° desfasado en el círculo. Esto pide
#     una zanahoria MÁS LARGA en curva, como anticipación.
#
# Cuál gana depende del retardo real de TU dron, y no lo puedo simular:
# mi modelo da 0.25 cm de error donde tú mides 4.11, así que no reproduce
# lo que sea que domine. Es una pregunta empírica, no de sobremesa.
#
#   1.0 = zanahoria constante (lo que volaste, 4.96 cm RMS)
#   <1  = más corta en curva  (ataca el recorte por cuerda)
#   >1  = más larga en curva  (ataca el retardo)
#
# Vuela 0.7 y 1.4 y quédate con el que dé menos RMS. Son dos vuelos y
# zanja la discusión.
AVANCE_CURVA_K = 1.0
AVANCE_CURVA_REF = 2.0   # 1/m de curvatura donde el efecto es pleno
                         # (kappa 2.0 = radio 0.5 m)

# Perro guardián: si la posición del MoCap se queda vieja, se deja de
# avanzar la zanahoria y se repite la última consigna. El dron se queda
# flotando ahí en vez de seguir a ciegas.
MOCAP_TIMEOUT = 0.5    # s (Robotat a 30 Hz desde oct-2026: 0.25 cortaba con huecos normales)
# Cuánto se tolera SIN MoCap antes de bajar a ciegas. Por debajo de esto el
# dron se queda flotando en la última consigna, que es lo correcto para un
# hueco corto de WiFi. Por encima, el dron está en el aire sin saber dónde
# está y flotar solo gasta batería: se baja.
MOCAP_ATERRIZA_S = 1.5
TOL_META      = 0.10   # m de arco restante para dar la ruta por hecha
FACTOR_TIMEOUT = 2.5

# ══════════════════════════════════════════════════════════════
# GIRO DE LA NARIZ
# ══════════════════════════════════════════════════════════════
SEGUIR_YAW_RUTA = True    # False → nariz fija (más preciso, pero la cámara
                          #         no encuadra nada en concreto)

# HACIA DÓNDE MIRA LA CÁMARA respecto al avance:
#
#   'tangente'      : hacia adelante, en la dirección de la marcha.
#   'perpendicular' : a 90° del avance. Si el dron va de derecha a
#                     izquierda, la cámara mira hacia el lado. En el
#                     contorno esto hace que mire SIEMPRE HACIA AFUERA,
#                     porque el lado se elige a partir del sentido en que
#                     se recorre la ruta.
#
# 12-sep-2026: cambiado a 'tangente'. La cámara tiene que ver hacia dónde
# va el dron para poder esquivar lo que tenga delante; mirando a 90° el
# encuadre cubre de 55° a 125° del avance y el frente queda ciego.
# Esto NO cambia casi nada del vuelo: con el montaje mal, 'perpendicular'
# ya dejaba la nariz a -5.2° de la tangente. El cambio real está en que
# ahora el código SABE que mira al frente, que es lo que arregla la
# proyección de los obstáculos.
# Para volver a cobertura de cultivo mirando al lado: 'perpendicular'.
YAW_MODO = 'tangente'

# De qué lado del avance mira, en modo perpendicular:
#
#   'auto'      : lo deduce de la ruta. Se calcula el área con signo
#                 (fórmula del zapatero) del recorrido: si va en sentido
#                 antihorario, "afuera" es la DERECHA del avance; si va en
#                 horario, la IZQUIERDA. Verificado sobre un contorno
#                 circular en los dos sentidos: la normal apunta hacia
#                 afuera en ambos casos.
#   'izquierda' / 'derecha' : forzado, por si la ruta no encierra un área
#                 clara y 'auto' se equivoca.
YAW_LADO = 'auto'

# Ajuste fino del ángulo, por si 90° exactos no encuadran bien el cultivo.
# 90 = perpendicular puro. 70 = perpendicular pero algo adelantado.
YAW_ANGULO_DEG = 90.0

# EL PROBLEMA DEL MARCO, que hay que resolver antes de mandar ningún yaw:
#
# `send_position_setpoint(x, y, z, yaw)` toma el yaw EN EL MARCO DEL EKF.
# Pero la tangente de la ruta está en el marco del MUNDO (el del MoCap).
# Y no coinciden: el EKF llama "0°" a la dirección en la que miraba el dron
# cuando se reseteó el estimador. Medido en sus dos vuelos:
#
#     yaw EKF   yaw MoCap   yaw MUNDO real   desfase
#      -0.7°     -168.7°        +11.3°       +12.0°
#      -2.0°     -169.8°        +10.2°       +12.2°
#
# Si le mando yaw=0 pensando en el eje X del mundo, el dron apunta a otro
# sitio. Y como NO mandamos extpose (fue lo que rompió la v2), el yaw del
# EKF además DERIVA con el giroscopio: el desfase ni siquiera es constante
# durante el vuelo.
#
# SOLUCIÓN: medirlo en marcha. En cada ciclo,
#     desfase = yaw_mundo_del_MoCap − yaw_del_EKF
# filtrado despacio, y se resta antes de mandar. Con eso el lazo de yaw
# queda cerrado contra el MoCap SIN tocar el estado interno del EKF, que
# es exactamente lo que queremos: la precisión del MoCap sin el riesgo de
# send_extpose.
YAW_MONTAJE_DEG = 5.2     # ✓ CONFIRMADO 17-sep-2026 con dos vuelos.
                          #
                          # ═══ CÓMO SE LLEGÓ A ESTE NÚMERO ═══
                          # El error de montaje no aparece en NINGUNA medida
                          # que haga el propio lazo, porque el lazo fuerza
                          # (yaw_mocap − M) → rumbo y M se cancela. La nariz
                          # física apunta a:
                          #
                          #     nariz_real = rumbo + (M_supuesto − M_real)
                          #
                          # Así que hace falta una observación EXTERNA: mirar
                          # el dron. Con la ruta `Linea_recta`, que va por
                          # X=0 con rumbo −90° a la ida y +90° a la vuelta:
                          #
                          #   vuelo 1, M=275.2 → nariz vista en +X (0°)
                          #            0 = 90 + (275.2 − M_real) → M_real=5.2
                          #   vuelo 2, M=185.2 → nariz vista AL REVÉS (+180°)
                          #            180 = 185.2 − M_real      → M_real=5.2
                          #
                          # Dos observaciones distintas, el mismo número.
                          #
                          # Y una comprobación INDEPENDIENTE que no usa nada
                          # de lo anterior: el desfase EKF↔mundo que mide el
                          # programa al arrancar pasó de +87.5° a +175.4°
                          # entre los dos vuelos, o sea +87.9°, justo cuando
                          # M cambió en −90°. Como desfase = yaw_mocap − M −
                          # yaw_EKF, un cambio de −90 en M tiene que dar +90
                          # en el desfase. Salió +87.9: los 2° que faltan son
                          # que el dron no se dejó exactamente igual sobre el
                          # suelo. El modelo es correcto.
                          #
                          # SI ALGÚN DÍA VUELVE A MIRAR MAL: no es el lazo.
                          # M es una propiedad de cómo está definido el
                          # cuerpo rígido en Motive, así que cambia si se
                          # redefine ahí o si se mueve la cámara en el dron.
                          # Se vuelve a medir en 30 s, sin volar:
                          #
                          #     python volar_ruta.py --calibrar
                          #
                          # ── HISTORIA, PARA NO REPETIRLA ──
                          # 180.0  estimado a ojo de dos vuelos (pre 12-sep)
                          # 275.2  "medido" el 12-sep con --calibrar. Estaba
                          #        90° fuera: o el cuerpo rígido se redefinió
                          #        después, o al calibrar la nariz apuntaba a
                          #        +Y creyendo que era +X. Es el error fácil
                          #        de cometer, y por eso --calibrar ahora
                          #        insiste tanto en CUÁL es el eje +X.
                          # 185.2  deducción intermedia del 17-sep, con la
                          #        rama equivocada de ±90°.
                          #   5.2  ESTE. Dos observaciones + una comprobación
                          #        independiente.
YAW_RATE_MAX    = 45.0    # °/s de la referencia. Simulado 20/45/90: entre
                          # 45 y 90 no cambia nada medible, y 45 es más
                          # suave para la cámara.
YAW_LOOKAHEAD   = 0.10    # m por delante de la zanahoria donde se lee la
                          # tangente: mira hacia donde VA a ir, no a donde
                          # ya está.
YAW_ALPHA_OFF   = 0.02    # filtro del desfase. Muy lento a propósito: el
                          # desfase real cambia despacio (deriva del giro),
                          # así que si se mueve rápido es ruido.
# BANDA MUERTA del acople nariz→velocidad. Por debajo de este error NO se
# frena nada.
#
# MEDIDO EN TU VUELO, y era un fallo de ajuste mío: el informe dijo
# "afloja el 97 % del tiempo, factor mínimo 0.55" con un error de nariz
# medio de 8.6°. Sin banda muerta, factor = 1 − 8.6/60 = 0.86, o sea que
# le estaba pisando el freno un 14 % de forma permanente por un error de
# 8 grados que no molesta a nadie. Cuadra con el tiempo: 47.2 s reales
# contra 31.8 estimados = 0.67×.
# Con banda de 15°, un error de 8.6° no frena nada en absoluto.
YAW_BANDA_MUERTA = 15.0   # °
YAW_ERR_AFLOJA  = 60.0    # ° POR ENCIMA de la banda muerta para llegar al
                          # suelo del factor
YAW_FACTOR_MIN  = 0.35    # nunca por debajo: es lo que garantiza que NO se
                          # pare esperando a la nariz.
YAW_FIJO_DEG    = 0.0     # usado solo si SEGUIR_YAW_RUTA = False
# Aceleración angular de la REFERENCIA de yaw (°/s²). Sin esto, `paso()`
# saltaba de 0 a YAW_RATE_MAX en un solo ciclo: la referencia arrancaba con
# un escalón de velocidad angular, el dron respondía con un tirón, y ese
# tirón es un pico de corriente en los cuatro motores a la vez. Con 120 °/s²
# la referencia tarda 0.37 s en llegar a 45 °/s, que no cambia nada medible
# del seguimiento y quita el escalón.
YAW_ACEL_MAX    = 120.0   # °/s²  (0 = sin límite, como antes)
# Margen de la curva de frenado: se frena con esta fracción de la
# aceleración disponible, para que el limitador pueda seguirla y no haya
# que cortar la referencia en el último ciclo. 0.5 sale del banco.
YAW_MARGEN_FRENO = 0.5

# ══════════════════════════════════════════════════════════════
# APROXIMACIÓN AL PRIMER PUNTO  (añadido 17-sep-2026)
# ══════════════════════════════════════════════════════════════
# Antes esto era una sola línea:
#
#     pos_cmd(path[0][0], path[0][1], altura)
#
# repetida a 50 Hz. O sea: un ESCALÓN de consigna de hasta un metro. El
# controlador de posición del dron ve un error de 1 m y hace lo único que
# sabe hacer, que es inclinarse al máximo y tirar. Ese es el arranque más
# violento de todo el vuelo —más que el despegue, que sí va limitado a
# 10 cm/s— y es un pico de corriente de los cuatro motores a la vez.
#
# El GAP8 del AI-deck comparte esa alimentación y es el que más corriente
# pide cuando captura. Con la LiPo cansada, ese pico es justo el momento en
# que se cuelga.
#
# Ahora la consigna RECORRE la recta desde donde está el dron hasta el
# primer punto, con el mismo perfil de jerk mínimo que usa la ruta
# (g(u) = 10u³ − 15u⁴ + 6u⁵): empieza con velocidad Y aceleración cero, y
# termina igual. No hay ningún escalón en ningún punto del trayecto.
#
# Los números salen de `probar_suavidad.py`, no a ojo. Con el perfil de
# jerk mínimo el pico de velocidad es 1.875× la media.
#
# Lo que de verdad cuesta corriente es la ACELERACIÓN, no la velocidad: es
# la que inclina el dron y la que obliga a los motores a dar un tirón. La
# velocidad de crucero, una vez alcanzada, casi no cuesta nada.
#
# Estos valores son los LENTOS (17-sep, segunda pasada). Un metro tarda
# 15.6 s y el pico de aceleración queda en 0.024 m/s², que son 0.14° de
# inclinación. La consigna de escalón que había antes pedía la inclinación
# máxima, del orden de 5 m/s²: son DOSCIENTAS veces más.
#
# ══════════════════════════════════════════════════════════════
#  ►►►  ESTE ES EL NÚMERO PARA IR MÁS RÁPIDO O MÁS LENTO AL PRIMER
#       PUNTO. Es el único que hay que tocar.  ◄◄◄
# ══════════════════════════════════════════════════════════════
# Medido con `probar_suavidad.py` sobre tu tramo real de 2.2 m:
#
#     APROX_V_MAX    tarda     pico de aceleración
#     ───────────    ─────     ───────────────────
#        0.12 m/s     34 s     0.011 m/s²   ← el del 17-sep, muy lento
#        0.25 m/s     17 s     0.047 m/s²   ← ESTE, el de ahora
#        0.35 m/s     13 s     0.080 m/s²   ← aquí manda ya APROX_A_MAX
#        0.50 m/s     13 s     0.080 m/s²   ← igual: subir más no hace nada
#
# Por debajo de 13 s el que limita deja de ser la velocidad y pasa a ser
# `APROX_A_MAX`, así que subir APROX_V_MAX por encima de 0.35 no cambia
# nada. Para ir aún más rápido habría que subir también APROX_A_MAX, y ahí
# es donde vuelve el tirón que colgaba al GAP8 — hazlo de 0.02 en 0.02 y
# comprueba la cámara, no de golpe.
APROX_V_MAX  = 0.25   # m/s  ← EL NÚMERO DE LA VELOCIDAD DE APROXIMACIÓN
APROX_A_MAX  = 0.08   # m/s² aceleración de punta — NO subir para ir rápido
APROX_T_MIN  = 3.0    # s    duración mínima, aunque el tramo sea corto
APROX_TOL    = 0.05   # m    se da por llegado dentro de este radio
APROX_T_EXTRA = 5.0   # s    margen de asentamiento tras el perfil
# Pausa quieta entre fases (despegue → aproximación → giro → ruta). No es
# estética: cada fase deja a los motores con una corriente distinta, y
# encadenarlas sin respiro suma los transitorios. Dos segundos quietos
# entre una y otra hacen que cada arranque parta siempre del mismo estado,
# que es de lo que depende la CONSISTENCIA.
APROX_PAUSA  = 2.0    # s

# Giro de alineación previo a la ruta. Mucho más lento que YAW_RATE_MAX a
# propósito: es un giro en el sitio, no hay ninguna prisa, y era el otro
# movimiento brusco. A 12 °/s, 90° tardan 7.5 s.
YAW_RATE_APROX  = 12.0   # °/s
YAW_ACEL_APROX  = 30.0   # °/s²  solo para este giro; la ruta sigue usando
                         # YAW_ACEL_MAX, para no volver perezosa la nariz
                         # en las vueltas en U
YAW_TOL_APROX   = 4.0    # °  se da por alineado con este error
YAW_T_MAX_APROX = 25.0   # s  tope, por si el lazo no converge


# ══════════════════════════════════════════════════════════════
# EVASIÓN DE OBSTÁCULOS
# ══════════════════════════════════════════════════════════════
#   'off'      : ni se mira la cámara.
#   'observar' : DETECTA, proyecta al mundo y apunta en el informe lo que
#                HABRÍA hecho, pero NO toca la ruta. ← empieza por aquí
#   'esquivar' : desvía la zanahoria de verdad.
#
# Empieza en 'observar' a propósito. En este proyecto, cada vez que he
# dado por bueno algo sin medirlo antes me he equivocado; y aquí lo que
# está sin medir es MiDaS, que no da metros y se equivoca. Un vuelo en
# 'observar' te dice cuántos obstáculos habría visto, dónde, y cuántos
# eran fantasmas, sin arriesgar el dron.
#
# ⚠ 16-sep-2026: PUESTO EN 'off' PORQUE ESTABA ARRUINANDO EL SEGUIMIENTO.
# No es una sospecha, está medido. En el vuelo de la línea recta, en
# modo 'observar' (o sea calculando pero sin aplicar):
#
#     ciclos con desvío : 1215 de 1620  =  75 % del vuelo
#     maniobras         : 2, las dos al TOPE de 40 cm
#     trípode detectado : NO — el obstáculo confirmado quedó a 1.12 m
#                         del trípode real
#
# En 'observar' todo eso se calculaba y se tiraba. En 'esquivar' se
# APLICA: la zanahoria se va hasta 40 cm de lado durante tres cuartos
# del vuelo, empujada por fantasmas. Eso aporta del orden de 22 a 35 cm
# de RMS, contra los 3.88 cm que da el seguimiento limpio. La evasión
# domina el error por un factor de 5 a 10, y por eso la trayectoria ya
# no se parece a la de antes.
#
# NO vuelvas a 'esquivar' hasta que EVASION_DISP_MIN esté puesto y un
# vuelo en 'observar' dé 1 maniobra con 1 trípode. Mientras la compuerta
# siga en None, 'esquivar' es un generador de ruido con forma de dron.
EVASION_MODO = 'esquivar'

# Campo de visión HORIZONTAL de la cámara del AI-deck, en grados.
# ⚠ VERIFÍCALO. El Himax HM01B0 con la óptica estándar ronda 65-90° según
# el lente que traiga. Si este número está mal, los obstáculos se
# proyectan en el sitio equivocado (el error crece con la distancia).
# Para medirlo: pon algo estrecho a 1 m justo en el borde del encuadre,
# mide su desplazamiento lateral d, y FOV = 2·atan(d/1.0).
EVASION_FOV_DEG = 70.0

EVASION_N_COL      = 9      # columnas en que se parte la imagen
EVASION_UMBRAL_REL = 1.25   # una columna es sospechosa si su disparidad
                            # supera esto × la mediana del cuadro.
                            # Señal RELATIVA: no depende de la calibración.

# ⚠ EL UMBRAL ANTERIOR, SOLO, NO SIRVE PARA DECIR "AQUÍ NO HAY NADA".
# En cualquier cuadro hay alguna columna por encima de 1.25 × la mediana,
# aunque mires a una pared lisa: basta el ruido de MiDaS. Medido en vuelo
# sobre el Robotat VACÍO, sin trípodes: 111 detecciones proyectadas y 748
# ciclos (37% del vuelo) en que se habría apartado al tope de 40 cm.
#
# Este segundo umbral compara contra la DISPERSIÓN del propio cuadro, no
# contra su mediana:  z = (disp − mediana) / (1.4826 · MAD) > EVASION_SIGMAS.
# En un cuadro plano el exceso crece igual que la dispersión, así que el z
# se queda bajo y no dispara. Un objeto de verdad se sale por muchos sigmas.
# Se usa mediana/MAD y no media/sigma porque el obstáculo está dentro de la
# muestra y con media/sigma se inflaría su propio umbral.
#
# CÓMO ELEGIRLO CON DATOS: el informe imprime los percentiles del z máximo
# por ciclo. Vuela en 'observar' SIN obstáculos y mira el p99: ese es el
# ruido de fondo de tu laboratorio. Ponlo por encima. Luego vuela CON el
# trípode y comprueba que lo supera holgadamente.
# None = desactivado (comportamiento anterior).
EVASION_SIGMAS = 3.0

# ── LA COMPUERTA QUE DE VERDAD HACE FALTA ──────────────────────────────
# Disparidad BRUTA mínima para considerar que una columna es un obstáculo.
# None = desactivada.
#
# POR QUÉ NO BASTA CON LO DE ARRIBA
# En el laboratorio real hay objetos a distintas distancias en distintas
# direcciones, y el detector los ve BIEN: destacan de verdad sobre el
# cuadro, así que pasan tanto el umbral relativo como el de sigmas. El
# fallo no está en detectarlos, está en A QUÉ DISTANCIA se colocan: la
# calibración del banco satura en 1.79 m, así que una pared a 3 m acaba
# anotada a 0.4 m, encima de la ruta. Eso explica que 0 de 111
# detecciones se descartaran por zona en el vuelo del Robotat vacío.
#
# Esta compuerta corta ANTES de convertir a metros, sobre el número que
# MiDaS da directamente, así que no hereda el error de la calibración.
#
# CÓMO FIJARLA, QUE ES LO ÚNICO HONESTO AQUÍ:
#   1. Vuela en 'observar' con el laboratorio COMO ESTARÁ, pero SIN los
#      trípodes. El informe imprime los percentiles de disparidad máxima
#      por cuadro: ese es el fondo de tu laboratorio.
#   2. Pon EVASION_DISP_MIN por encima del p99 de ese vuelo.
#   3. Vuela con los trípodes y comprueba en el informe que los superan.
# Si el trípode no supera el fondo, la cámara no puede distinguirlo y hay
# que acercar el umbral de decisión, no forzar el número.
EVASION_DISP_MIN = None

# ── AUTOCALIBRACIÓN DE LA COMPUERTA ────────────────────────────────────
# Si `EVASION_DISP_MIN` es None y esto está en True, la compuerta se mide
# SOLA en cada vuelo: mientras el dron está quieto estabilizándose sobre el
# punto inicial —ya son 1-2 s que hoy no se aprovechan para nada— se
# recogen cuadros y se toma
#
#     compuerta = percentil(disparidad máxima por cuadro, 99) · MARGEN
#
# QUÉ EXIGE: que en ese momento no haya nada delante del dron. Es el mismo
# requisito que el vuelo de medición a mano, pero sin gastar un vuelo.
# QUÉ DA A CAMBIO: se adapta a la luz y al fondo de ESE día. La disparidad
# de MiDaS no es absoluta; el fondo de tu laboratorio cambia con la
# iluminación, así que un número fijado hace una semana puede no valer hoy.
#
# Un valor puesto A MANO en EVASION_DISP_MIN MANDA siempre sobre esto.
EVASION_DISP_AUTO = True
EVASION_DISP_AUTO_SEG = 2.0      # intento ANTES de arrancar, en segundos
EVASION_DISP_AUTO_MARGEN = 1.15  # 15 % por encima del p99 del fondo
EVASION_DISP_AUTO_MIN_N = 10     # cuadros mínimos para fiarse del fondo

# ⚠ LO ANTERIOR NO BASTA CON ESTA CÁMARA, y es un error que ya cometí.
# La primera versión pedía 12 cuadros en los 2 s de estabilización. El
# vuelo del 17-sep dio 2 cuadros: el AI-deck en RAW va a 0.3–1.1 fps, no a
# 10. Con 0.76 fps de media, 12 cuadros son 16 segundos de dron quieto en
# el aire, que es inaceptable.
#
# Así que si antes de arrancar no se consiguen, se SIGUE MIDIENDO DURANTE
# LA RUTA, que es gratis. Mientras la compuerta no esté puesta NO se mete
# nada en el mapa: las detecciones de ese rato solo sirven para medir el
# fondo. Un obstáculo que aparezca antes de tenerla se pierde, y es lo
# correcto: sin compuerta no se puede distinguir de ruido.
#
# ESTO EXIGE que el principio de la ruta esté despejado. Si el trípode
# está en mitad del recorrido, se cumple solo.
EVASION_DISP_AUTO_EN_VUELO = True

# Fallos SEGUIDOS procesando la cámara tras los que se apaga la evasión.
# Antes esto era un `except Exception: pass` y no se enteraba nadie.
EVASION_ERRORES_MAX = 30

# Velocidad máxima a la que puede MOVERSE el desvío lateral, en m/s.
# Sin esto el desvío saltaba de 0 a 40 cm en un ciclo de 20 ms, y de +40 a
# −40 cm si cambiaba de lado: de ahí el tirón brusco en vuelo.
#
# ES ASIMÉTRICO, Y ESA ES LA CLAVE
# Con un solo límite hay que elegir entre dos cosas que no se llevan:
# apartarse rápido (llegar a tiempo) y volver despacio (no dar tirones).
# Medido: con 0.30 m/s simétrico la rampa se comía 1.33 s = 13 cm de
# recorrido, que se SUMAN a los 30 cm que tarda `fuerza` en llegar a 1.
# Resultado: detectando a 30 cm, el desvío solo estaba completo a 11 cm.
#
# Pero las dos direcciones no tienen el mismo riesgo. Alejarse del
# obstáculo tarde es un choque; volver a la ruta rápido solo es un tirón.
# Así que se separan: SALIDA rápida, REGRESO suave.
#
#   SALIDA 0.60 m/s → tope de 40 cm en 0.67 s (antes 1.33 s)
#   REGRESO 0.30 m/s → sigue sin dar el tirón de vuelta
#
# 0.60 m/s de consigna lateral el dron la sigue de sobra: con 10° de
# inclinación acelera a 1.7 m/s², o sea que alcanza esa velocidad en
# 0.35 s. Si aun así llega tarde, esta es la primera que hay que subir.
EVASION_DESVIO_RATE = 0.30        # REGRESO a la ruta (suave)
EVASION_DESVIO_RATE_SAL = 0.60    # SALIDA hacia fuera (rápida)

# ── TRES FILTROS PARA NO CONFUNDIR EL LABORATORIO CON UN OBSTÁCULO ──
#
# 1) FRANJA VERTICAL de la imagen que se mira.
#    Con la cámara a la altura del dron y mirando horizontal, la mitad
#    inferior del cuadro es suelo. Subiendo el borde inferior se quita de
#    un plumazo. (0.20, 0.70) mira sobre todo a la altura del dron y por
#    encima, que es lo que le puede pegar.
EVASION_FRANJA = (0.20, 0.70)

# 2) TEST DE VERTICALIDAD. El suelo se ACERCA según bajas por la imagen
#    (disparidad = tan(θ)/h), un poste está a la misma distancia en todas
#    sus filas. Se compara la mitad inferior de la franja con la superior:
#       suelo   → ratio ≈ 10
#       trípode → ratio ≈ 1.0
#    Medido sobre escenas sintéticas con ruido: 10× de separación.
#    Por encima de este valor, la columna se descarta por ser suelo.
EVASION_MAX_RATIO = 1.6

# Medido: percentil 90 sobre columnas anchas no ve un poste delgado pasados
# ~0.6 m; con Depth Anything el 97 lo ve hasta 0.8-1.0 m; con MiDaS no ayuda.
EVASION_PERCENTIL_POR_MODELO = {'midas': 90.0, 'depth_anything_v2': 97.0, 'depth_anything_v2_metrico': 90.0, 'hibrido': 97.0}
EVASION_PERCENTIL = EVASION_PERCENTIL_POR_MODELO.get(VISION_MODELO, 90.0)

# 3) FILTRO POR ZONA. La estructura del Robotat (marco, patas, borde de la
#    mesa) es un objeto REAL y VERTICAL: el test de verticalidad no la
#    quita. Lo que la quita es saber que está donde no vuelas.
#    Solo se aceptan detecciones a menos de esta distancia de la ruta
#    planificada. Con eso caen las paredes, la estructura del laboratorio
#    y la gente mirando, sin configurar nada.
#    None = desactivado.
EVASION_DIST_RUTA = 0.80

# Y si algo concreto se sigue colando, se excluye a mano:
#   [(x, y, radio), ...] en coordenadas del laboratorio.
EVASION_ZONAS_PROHIBIDAS = []
EVASION_DIST_MAX   = 2.0    # m: más lejos no interesa (y MiDaS es peor)
EVASION_VIDA_MEDIA = 2.0    # s: olvido del mapa. Corto a propósito, para
                            # que un falso positivo no desvíe el resto del
                            # vuelo.
# Visto al menos 2 veces: con olvido continuo, a 0.5 s suman 1.84, nunca 2.0.
# Con vida media de 2 s: 1 + 0.5**(dt/2) >= 1.5 si dt <= 2 s en el mismo punto.
EVASION_PESO_MIN   = 1.5

EVASION_RADIO_INF  = 0.60   # m: desde aquí empieza a apartarse
EVASION_RADIO_SEG  = 0.30   # m: aquí el desvío ya es máximo
# ══════════════════════════════════════════════════════════════
# HOLGURA: a cuánto hay que pasar del punto detectado (17-sep-2026)
# ══════════════════════════════════════════════════════════════
# Esto es nuevo, y es el arreglo de "lo ve, se aparta, y aun así choca".
# Antes no existía ningún objetivo de separación: el desvío salía de un
# campo de potencial normalizado y acababa donde acabara. Medido sobre un
# racimo de cinco puntos —que es lo que da un trípode de verdad, no un
# punto limpio—, el desvío que producía era de OCHO CENTÍMETROS, porque
# los puntos a un lado y al otro de la ruta se cancelaban en la media.
# Ocho centímetros es rozarlo.
#
# La holgura se mide del punto detectado al CENTRO del dron, así que hay
# que sumar tres cosas:
#     radio del obstáculo ... 0.30 m  (las patas de un trípode se abren
#                                      ~30 cm desde donde MiDaS ve el
#                                      cuerpo central)
#     radio del dron ........ 0.10 m  (con hélices)
#     margen ................ 0.10 m  (error de proyección de la cámara)
#                             ───────
#                             0.50 m
#
# Si tus trípodes son más anchos o el error de proyección es mayor, sube
# esto. El tope `EVASION_DESVIO_MAX` tiene que quedar POR ENCIMA, o la
# holgura no se puede cumplir y el dron pasa más cerca de lo pedido.
#
# ══ AJUSTADO 18-sep-2026 CON MEDIDA DE VUELO ══
# Con `EVASION_HOLGURA = 0.50` pasaba a 35–40 cm medidos del trípode. O
# sea que la separación que se ve en el laboratorio sale en torno al 75 %
# de la holgura pedida: el punto que detecta MiDaS no es el centro del
# trípode, y además el dron va algo por detrás de su consigna.
#
#     separación real ≈ 0.75 × EVASION_HOLGURA
#
# Para los 20–25 cm pedidos → 0.30. Es un margen ajustado: quedan ~15 cm
# entre las hélices y las patas. Si lo ves demasiado justo, 0.35 da unos
# 26 cm. Este es EL número que hay que tocar para separarse más o menos.
EVASION_HOLGURA    = 0.43   # m
EVASION_DESVIO_MAX = 0.60   # m: tope del desvío lateral. Tiene que quedar
                            # holgadamente POR ENCIMA de EVASION_HOLGURA:
                            # si se queda justo, cualquier racimo un poco
                            # ancho pide más de lo que cabe, eso se declara
                            # "no cabe", y "no cabe" es una PARADA.
                            # OJO con el geofence: este desvío se suma a
                            # la ruta, así que la ruta tiene que caber en
                            # el área con este margen a los lados.

# ══════════════════════════════════════════════════════════════
# RODEAR, NO HACER ESES  (18-sep-2026)
# ══════════════════════════════════════════════════════════════
# Tres números que convierten "se aparta" en "lo rodea":
#
# COLA: cuánto se sigue empujando DESPUÉS de que el obstáculo quede por
#   detrás de la ZANAHORIA. La zanahoria va por delante del dron, así que
#   cuando ella ya pasó el trípode, el dron todavía está a su altura.
#   Soltar el desvío ahí es volver a la ruta rozándolo. Con 35 cm de cola
#   el dron termina de pasar antes de que la consigna vuelva.
#
# COMPROMISO: a partir de qué desvío se fija el lado, y cuánto se mantiene.
#   Sin esto el lado se reelegía 50 veces por segundo y el dron hacía eses
#   delante del obstáculo sin pasarlo nunca. El tope de tiempo es solo una
#   red de seguridad: a V_MAX=0.10 m/s, rodear un trípode son unos 8 s.
EVASION_COLA             = 0.35   # m
EVASION_COMPROMISO_MIN   = 0.05   # m: desvío desde el que se fija el lado
EVASION_COMPROMISO_MAX_S = 20.0   # s: tope de seguridad del compromiso
EVASION_CICLOS_SOLTAR    = 10     # ciclos seguidos a cero para soltar
EVASION_MARGEN_LADO = 0.05       # m: ventaja necesaria para elegir derecha
EVASION_VISTA_MAX_S = 1.0        # edad máxima de una imagen para maniobrar
MANIOBRA_BLOQUEO_ATERRIZA_S = 3.0  # segundos en BLOQUEADA antes de aterrizar
MANIOBRA_ESPERA_VISTA_S = 3.0
BATERIA_ATERRIZA_S = 5.0          # segundos seguidos con batería crítica
EVASION_CAL_MODELO = 'hibrido'     # modelo al que pertenecen CAL_A/B y DISP_MIN
EVASION_CAL_FORMATO = 'raw'  # formato de la cámara con el que se midieron CAL_A/B y DISP_MIN: raw o jpeg
EVASION_RADIO_PARO = 0.22   # m: más cerca que esto y justo delante → se
                            # queda flotando en vez de seguir
EVASION_ANCHO_PARO = 0.18   # m: medio ancho del corredor que mira el paro

# Calibración disparidad → metros. Se obtiene con:
#     python volar_ruta.py --calibrar-midas
# Sin esto, la evasión funciona igual pero SOLO con la señal relativa: sabe
# que hay algo cerca y en qué dirección, no a cuántos metros.
# Medido 12-sep-2026 con trípode a 0.30 / 0.40 / 0.60 m.
#   R² = 0.9997, error medio 0.2 cm.  Pendiente a entre puntos
#   consecutivos: 0.007172 y 0.007610 — consistente al 6%, que es lo que
#   dice que el modelo físico se cumple.
#
# RANGO DE VALIDEZ: 0.3 a 0.6 m. Cubre los tres umbrales que deciden
#   (RADIO_INF 0.60, RADIO_SEG 0.30, RADIO_PARO 0.22), así que sirve para
#   lo que importa. Fuera de ahí extrapola.
#
# OJO CON EL TECHO: los tres puntos están apiñados cerca, así que `a` queda
#   bien determinado pero `b` no, y `b` es quien manda en lo lejano. El
#   modelo satura en 1/b = 1.79 m: nunca devuelve más que eso. Como b > 0
#   no hay singularidad ni metros negativos, o sea que el fallo es por
#   saturación, que es el lado seguro. Pero EVASION_DIST_MAX = 2.0 queda
#   INERTE: ninguna estimación llega a 2 m, así que ese filtro no descarta
#   nada. Los que siguen filtrando son EVASION_UMBRAL_REL, el test de
#   verticalidad y EVASION_DIST_RUTA.
#
# PARA MEJORARLO: repetir con la dirección despejada y 5-6 puntos hasta
#   1.5 m. Eso fija b y quita el techo. La calibración de arriba se hizo
#   con objetos alrededor y el punto de 0.80 m midió uno fijo a 0.57 m en
#   vez del trípode; sobre la base del Robotat no hay ese problema.
EVASION_CAL_A = 0.003186
EVASION_CAL_B = 0.508963


def _wrap(a):
    """Ángulo a (-pi, pi]. Sin esto, un giro de 179° a -179° se lee como
    358° y el dron da la vuelta entera por el lado largo."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _yaw_mundo_del_mocap():
    """
    Hacia dónde mira el dron DE VERDAD, en el marco del mundo.

    `mocap_pose['yaw']` es el yaw del cuerpo rígido definido en OptiTrack,
    que no coincide con el frente físico del dron: hay que quitarle el
    desfase de montaje. Devuelve None si el puente no manda orientación.
    """
    y = P.mocap_pose.get('yaw', None)
    if y is None or P._mocap_ori is None:
        return None
    return _wrap(float(y) - np.radians(YAW_MONTAJE_DEG))


def _yaw_ekf(scf):
    """Yaw que cree tener el dron, en radianes. None si no hay log aún."""
    try:
        return _wrap(np.radians(float(scf.pose_data['yaw'])))
    except Exception:
        return None


def _alinear_yaw_ekf(scf):
    """
    Reinicia el EKF con su yaw inicial = el rumbo real del dron (según el
    MoCap), para que no importe cómo se puso el dron. Tras un reset el EKF
    arranca en el yaw `kalman.initialYaw` (rad) con roll/pitch a 0, igual que
    siempre: solo cambia el yaw, no se usa el roll/pitch del MoCap.
    Con el dron en el suelo. True si se aplicó; el chequeo posterior decide.
    """
    ym = _yaw_mundo_del_mocap()
    if ym is None:
        print('  ⚠ Sin orientación del MoCap: no se puede alinear el yaw del EKF.')
        return False
    try:
        if 'initialYaw' not in scf.cf.param.toc.toc.get('kalman', {}):
            print('  ⚠ Este firmware no tiene kalman.initialYaw: el dron debe '
                  'ponerse con el rumbo del reset.')
            return False
        scf.cf.param.set_value('kalman.initialYaw', f'{ym:.6f}')
        scf.cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        scf.cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(ALINEA_YAW_CONV_S)
    except Exception as e:
        print(f'  ⚠ No se pudo alinear el yaw del EKF: {e}')
        return False
    print(f'  EKF reiniciado con yaw inicial {np.degrees(ym):+.1f}° '
          '(el rumbo real del dron).')
    return True


def _guardar_registro_despegue(reg):
    """Guarda el ascenso (t, x, y, z, roll, pitch, yaw del EKF) en
    capturas_vuelo/: si un despegue falla, ahí se ve por qué."""
    if not reg:
        return
    try:
        import json
        import os
        carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'capturas_vuelo')
        os.makedirs(carpeta, exist_ok=True)
        ruta = os.path.join(carpeta, time.strftime('despegue_%Y%m%d_%H%M%S.json'))
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump({'columnas': ['t_s', 'x', 'y', 'z', 'roll_ekf_deg',
                                    'pitch_ekf_deg', 'yaw_ekf_deg'],
                       'filas': reg}, f)
        print(f'  Registro del despegue: {ruta}')
    except Exception as e:
        print(f'  (no se pudo guardar el registro del despegue: {e})')


def _guardar_registro_maniobra(filas):
    """Guarda el estado y diagnóstico de la maniobra por ciclo de ruta."""
    if not filas:
        return
    try:
        import json
        import os
        carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'capturas_vuelo')
        os.makedirs(carpeta, exist_ok=True)
        ruta = os.path.join(carpeta, time.strftime('maniobra_%Y%m%d_%H%M%S.json'))
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump({'columnas': ['t_s', 'estado', 'x', 'y', 'z',
                                    'yaw_mocap_deg', 'yaw_obj_deg', 'fijo_x',
                                    'fijo_y', 'err_yaw_deg', 'err_pos_cm',
                                    'yaw_estable', 'edad_vista_s'],
                       'filas': filas}, f)
        print(f'  Registro de la maniobra: {ruta}')
    except Exception as e:
        print(f'  (no se pudo guardar el registro de la maniobra: {e})')


def _rampa_inicio(t, t0, duracion):
    """Factor smoothstep de arranque, limitado a [0, 1]."""
    if duracion <= 0:
        return 1.0
    u = max(0.0, min((t - t0) / duracion, 1.0))
    return u * u * (3 - 2 * u)


def _formato_camara(vis):
    """Formato más frecuente recibido; None mientras no haya frames."""
    receptor = vis.receptor
    if receptor is None:
        return None
    formatos = receptor.formatos
    if formatos.get(1, 0) > formatos.get(0, 0):
        return 'jpeg'
    if formatos.get(0, 0) > 0:
        return 'raw'
    return None


def _esperar_vision_lista(vis, espera_s, mapas_min=2):
    """Espera el modelo y cambios reales de mapa antes de permitir despegar."""
    if P.stop_event.is_set():
        return False
    t0 = time.perf_counter()
    t_print = t0
    mapa_anterior = vis.ultimo().get('mapa_ts')
    cambios = 0
    while not P.stop_event.is_set():
        ahora = time.perf_counter()
        if ahora - t0 >= espera_s:
            print('  No se despega: se agotó la espera del modelo de profundidad '
                  'y de mapas nuevos.')
            return False
        formato_listo = True
        if (EVASION_MODO in ('observar', 'esquivar')
                and VISION_MODELO != 'depth_anything_v2_metrico'):
            formato = _formato_camara(vis)
            formato_listo = formato is not None
            if formato_listo and formato != EVASION_CAL_FORMATO:
                print(f'  No se despega: la calibración es de {EVASION_CAL_FORMATO}, '
                      f'la cámara manda {formato}; recalibra con --calibrar-midas '
                      f'y pon EVASION_CAL_FORMATO = {formato!r}.')
                return False
        mapa_ts = vis.ultimo().get('mapa_ts')
        if vis.midas_listo:
            if mapa_ts is not None and mapa_ts != mapa_anterior:
                cambios += 1
            if cambios >= mapas_min and formato_listo:
                return not P.stop_event.is_set()
        else:
            cambios = 0
        mapa_anterior = mapa_ts
        if ahora >= t_print:
            print(f'  Esperando al modelo de profundidad, {ahora - t0:.0f} s '
                  f'({cambios}/{mapas_min} cambios de mapa)')
            t_print = ahora + 5.0
        P.stop_event.wait(min(0.05, max(0.0, espera_s - (ahora - t0))))
    return False


def _esperar_mocap_vivo(espera_s, confirmar_s=0.5):
    """
    True cuando la pose del MoCap lleva `confirmar_s` seguidos cambiando (el
    ruido real de Motive la mueve siempre, aunque el dron esté quieto). False
    si en `espera_s` no lo consigue: Motive no ve el dron en ese sitio y lo
    que llega es la última pose repetida.
    """
    t0, t_vivo = time.time(), None
    while time.time() - t0 < espera_s and not P.stop_event.is_set():
        if time.perf_counter() - P._t_mocap_cambio[0] < P.MOCAP_CONGELADO_S:
            if t_vivo is None:
                t_vivo = time.time()
            elif time.time() - t_vivo >= confirmar_s:
                print('  MoCap rastreando el dron.')
                return True
        else:
            t_vivo = None
        time.sleep(0.02)
    if not P.stop_event.is_set():
        print('  ✗ El MoCap no ve al dron en este sitio: repite la misma posición '
              'sin cambios (la última que vio).')
        print('    No se despega. Muévelo a un punto donde Motive lo vea (en '
              'Motive el cuerpo rígido debe aparecer rastreado con el dron en '
              'el suelo) y vuelve a lanzar.')
    return False


def _esperar_yaw_alineado(scf, tol_deg, espera_s):
    """
    True si el yaw del EKF coincide con el rumbo real del dron (o no se puede
    medir). False si tras `espera_s` sigue desalineado: con el dron en el suelo
    eso significa que despegaría corrigiendo torcido.
    """
    if _yaw_mundo_del_mocap() is None:
        print('  ⚠ El MoCap no manda orientación: no se puede verificar el yaw '
              'del EKF. Pon el dron con el rumbo del reset (0°).')
        return True
    t0, dif = time.time(), None
    while time.time() - t0 < espera_s and not P.stop_event.is_set():
        ye, ym = _yaw_ekf(scf), _yaw_mundo_del_mocap()
        if ye is not None and ym is not None:
            dif = float(np.degrees(_wrap(ym - ye)))
            if abs(dif) <= tol_deg:
                print(f'  Yaw del EKF alineado con el MoCap (Δ {dif:+.1f}°).')
                return True
        time.sleep(0.1)
    if dif is None:
        print('  ✗ No llegó el yaw del EKF: no se despega.')
    else:
        print(f'  ✗ Yaw del EKF desalineado con el MoCap: Δ {dif:+.1f}° '
              f'(máx {tol_deg:.0f}°). No se despega: el dron corregiría torcido.')
        print('    Si acabas de encender, espera unos segundos y repite; si '
              'persiste, revisa que el MoCap mande el cuaternión.')
    return False


def _esperar_ekf_listo(scf, espera_min_s, espera_max_s, tol_m, estable_s):
    """Exige posición EKF próxima al MoCap durante un intervalo continuo."""
    inicio = time.perf_counter()
    estable_desde = None
    proximo_aviso = inicio
    while not P.stop_event.is_set():
        ahora = time.perf_counter()
        pose = getattr(scf, 'pose_data', None) or {}
        mocap = P.mocap_pose.copy()
        dif = [pose[eje] - mocap[eje] if pose.get(eje) is not None
               and mocap.get(eje) is not None else None for eje in ('x', 'y', 'z')]
        texto = ' / '.join('sin datos' if d is None else f'{100 * d:+.1f}'
                           for d in dif)
        if ahora - inicio >= espera_max_s:
            print(f'  ✗ Tiempo de espera del EKF agotado: no se despega. '
                  f'Diferencia EKF-MoCap x/y/z = {texto} cm.')
            return False
        if ahora - inicio >= espera_min_s and all(
                d is not None and abs(d) <= tol_m for d in dif):
            if estable_desde is None:
                estable_desde = ahora
            if ahora - estable_desde >= estable_s:
                print(f'  EKF listo: diferencia con el MoCap x/y/z = {texto} cm')
                return not P.stop_event.is_set()
        else:
            estable_desde = None
        if ahora >= proximo_aviso:
            print(f'  Esperando EKF: diferencia EKF-MoCap x/y/z = {texto} cm')
            proximo_aviso = ahora + 2.0
        P.stop_event.wait(min(0.02, max(0.0, espera_max_s - (ahora - inicio))))
    return False


class ControlYaw:
    """
    Convierte "quiero mirar en esta dirección del MUNDO" en el yaw que hay
    que meter en send_position_setpoint, que va en el marco del EKF.

    Mantiene el desfase entre los dos marcos midiéndolo continuamente, así
    que la deriva del giroscopio no se acumula.
    """

    def __init__(self):
        self.desfase = None       # rad, mundo − EKF
        self.ref = None           # referencia de yaw en el mundo, suavizada
        self.vel = 0.0            # rad/s de la referencia (para limitar
                                  # la ACELERACIÓN angular, no solo la
                                  # velocidad)
        self.activo = False
        self.n_sin_dato = 0

    def calibrar(self, scf, intentos=30, mantener=None):
        """
        Mide el desfase antes de arrancar. False = no se puede girar.

        `mantener` se llama en cada vuelta y debe reenviar la consigna de
        posición. SIN ESO, esta función pasaba 1.5 s sin mandar nada: el
        commander del firmware tiene su propio watchdog de medio segundo,
        así que el dron se quedaba sin consigna a mitad de la maniobra y
        salía de ahí con un tirón. Era uno de los movimientos bruscos que
        se veían antes de arrancar la ruta.
        """
        vals = []
        for _ in range(intentos):
            if mantener is not None:
                mantener()
            ym, ye = _yaw_mundo_del_mocap(), _yaw_ekf(scf)
            if ym is not None and ye is not None:
                vals.append(_wrap(ym - ye))
            time.sleep(0.05)
        if len(vals) < 5:
            print('  ⚠ No se puede seguir la ruta con la nariz: el MoCap no')
            print('    manda orientación, o el log del dron no responde.')
            print('    Se vuela con el yaw FIJO. La ruta no se ve afectada.')
            return False
        # Mediana circular: robusta a un dato suelto raro
        v = np.array(vals)
        c, s = np.cos(v).mean(), np.sin(v).mean()
        self.desfase = float(np.arctan2(s, c))
        disp = float(np.degrees(np.std(_wrap(v - self.desfase))))
        print(f'  Desfase EKF↔mundo medido: {np.degrees(self.desfase):+.1f}° '
              f'(dispersión {disp:.1f}° en {len(vals)} muestras)')
        if disp > 15.0:
            print('  ⚠ Dispersión alta: el yaw del MoCap viene ruidoso.')
            print('    Se vuela con el yaw FIJO por seguridad.')
            return False
        self.activo = True
        return True

    def actualizar_desfase(self, scf):
        """Sigue la deriva del giroscopio. Filtro muy lento a propósito."""
        ym, ye = _yaw_mundo_del_mocap(), _yaw_ekf(scf)
        if ym is None or ye is None:
            self.n_sin_dato += 1
            return
        d = _wrap(ym - ye)
        self.desfase = _wrap(self.desfase
                             + YAW_ALPHA_OFF * _wrap(d - self.desfase))

    def fijar(self, yaw_mundo):
        """
        Planta la referencia donde se le diga, sin rampa y sin movimiento:
        se usa para arrancar el lazo YA alineado, en vez de dejar que la
        primera llamada a `paso()` lo descubra. Antes, al entrar en la ruta
        la referencia se inicializaba sola y el dron se recolocaba: era el
        tercer giro.
        """
        self.ref = _wrap(float(yaw_mundo))
        self.vel = 0.0

    def paso(self, yaw_deseado_mundo, dt, rate_max=None, acel_max=None):
        """
        Devuelve (yaw_a_mandar_en_grados, error_de_nariz_en_rad).

        La referencia se limita en velocidad angular —girar de golpe
        arrastra la posición y desestabiliza— y, desde el 17-sep, TAMBIÉN
        en aceleración angular. La diferencia importa: con solo el límite
        de velocidad, la referencia pasaba de 0 a 45 °/s entre dos ciclos
        consecutivos, que para el dron es un escalón igual de brusco que no
        tener límite ninguno. Ahora la velocidad angular sube y baja por
        rampa, así que el giro arranca y termina suave.

        `rate_max` (°/s) permite pedir un giro más lento que el de la ruta,
        que es lo que hace la alineación previa al despegue de la ruta.
        """
        if self.ref is None:
            self.ref = yaw_deseado_mundo
            self.vel = 0.0
        w_max = np.radians(YAW_RATE_MAX if rate_max is None else rate_max)
        a_cfg = YAW_ACEL_MAX if acel_max is None else acel_max
        err = _wrap(yaw_deseado_mundo - self.ref)

        if a_cfg > 0:
            a_max = np.radians(a_cfg)
            # Tres topes, y manda el más bajo:
            #  · w_max      : la velocidad angular que se permite.
            #  · sqrt(2a|e|): entrar solo tan rápido como se pueda frenar.
            #  · |e|/T      : el acercamiento final, para no tener que
            #                 cortar el último ciclo a cuchillo.
            # Curva de frenado con MARGEN: se frena con la mitad de la
            # aceleración disponible (YAW_MARGEN_FRENO). Sin margen, la
            # curva pide exactamente `a_max` para frenar, el limitador de
            # aceleración no puede seguirla, la referencia llega al final
            # todavía con velocidad y hay que cortarla a cuchillo en el
            # último ciclo. Medido: 87 °/s² de pico con el límite en 30.
            # Con la mitad, el pico medido cae a 30 °/s² clavados y el
            # tiempo de giro no cambia (7.4 s en los dos casos).
            v_obj = np.sign(err) * min(
                w_max, np.sqrt(2.0 * a_max * YAW_MARGEN_FRENO * abs(err)))
            self.vel += np.clip(v_obj - self.vel, -a_max * dt, a_max * dt)
        else:
            self.vel = np.clip(err / dt, -w_max, w_max)

        paso = self.vel * dt
        if abs(paso) > abs(err):        # red de seguridad: nunca pasarse
            paso = err
            self.vel = paso / dt
        self.ref = _wrap(self.ref + paso)
        ym = _yaw_mundo_del_mocap()
        e_nariz = _wrap(self.ref - ym) if ym is not None else 0.0
        return float(np.degrees(_wrap(self.ref - self.desfase))), float(e_nariz)


def _lado_hacia_afuera(path):
    """
    ¿A qué lado del avance queda "afuera"?

    Se usa el área con signo del recorrido (fórmula del zapatero). Si la
    ruta se recorre en sentido ANTIHORARIO el área sale positiva, y la
    normal que apunta hacia afuera es la de la DERECHA del avance (girar
    la tangente −90°). En sentido horario, al revés.

    Verificado sobre un contorno circular en los dos sentidos: en ambos
    la normal resultante apunta hacia afuera del círculo.

    Devuelve −1 (derecha del avance) o +1 (izquierda).
    """
    x, y = path[:, 0], path[:, 1]
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return (-1 if area > 0 else +1), area


def _yaw_objetivo(T, lado):
    """Dirección a la que debe mirar la nariz, dada la tangente."""
    rumbo = np.arctan2(T[1], T[0])
    if YAW_MODO == 'perpendicular':
        return _wrap(rumbo + lado * np.radians(YAW_ANGULO_DEG))
    return rumbo


# El objeto de visión lo crea main() como variable LOCAL, así que desde
# aquí no se puede alcanzar. Se envuelve la clase para quedarse con la
# referencia en cuanto se construya. No modifica el archivo base.
_VISION = {'obj': None}
_SistemaVisionBase = P.SistemaVision


class _SistemaVisionCapturado(_SistemaVisionBase):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _VISION['obj'] = self


P.SistemaVision = _SistemaVisionCapturado


class _MidiendoFondo(Exception):
    """
    Señal interna: este cuadro se usó para medir el fondo, no para detectar.

    Es una excepción y no un `return` porque el bloque de detección está
    dentro de un try/except que ya distingue fallos; salir por aquí deja
    clarísimo en el código que NO se alimentó el mapa, y el `except` de
    esta clase concreta impide que se cuente como avería de la cámara.
    """


class Evasion:
    """
    Une la cámara con la zanahoria: lee el mapa de MiDaS, proyecta lo que
    ve a coordenadas del mundo y devuelve cuánto hay que apartarse.

    Se apoya en `evasion.py`, que es donde está la geometría probada.
    """

    def __init__(self, vision):
        self.vision = vision
        self.activa = EVASION_MODO in ('observar', 'esquivar')
        self.esquiva = (EVASION_MODO == 'esquivar')
        if VISION_MODELO == 'depth_anything_v2_metrico' and self.esquiva:
            self.esquiva = False
            print('  [evasión] Modelo métrico forzado a OBSERVAR: en vuelo real no distinguió cerca de lejos con la cámara del AI-deck (casi choca con el trípode, 25-sep-2026).')
            print("  Para esquivar, usa VISION_MODELO = 'midas'.")
        self.mapa = None
        self.cal = None
        self.rumbos = None
        self.ts_visto = 0.0
        self.n_det = 0
        self.n_desvios = 0
        self.n_suelo = 0        # columnas descartadas por parecer suelo
        self.n_fuera = 0        # detecciones descartadas por estar lejos
        self.ruta_xy = None     # se rellena en preparar_ruta()
        self.n_paradas = 0
        self.n_no_cabe = 0    # ciclos sin maniobra posible (hueco estrecho)
        # Compromiso de lado: ver el bloque largo en paso().
        self.lado_comp = 0        # 0 = libre, +1 izquierda, -1 derecha
        self.mantener_compromiso = False
        self.vista = None
        self.t_comp = None
        self._ciclos_libre = 0
        self.n_compromisos = 0    # maniobras que llegaron a fijar lado
        self.n_comp_vencidos = 0  # las que se soltaron por tiempo
        self.desvio_max_visto = 0.0
        self.dist_min_vista = np.inf
        self.registro = []
        self._ev = None
        # Estado del limitador de pendiente del desvío (ver paso()).
        self.desvio_suave = 0.0
        self.t_desvio = None
        self.n_saturado = 0     # ciclos en que el limitador recortó
        self.n_errores = 0      # fallos SEGUIDOS procesando la cámara
        self.primer_error = None
        self.disp_fondo = None  # compuerta autocalibrada (ver calibrar_fondo)
        self.n_bajo_compuerta = 0
        self._fondo = []        # muestras de disparidad para la compuerta
        self.n_sin_compuerta = 0  # cuadros tirados por no tenerla todavía
        self.t_vuelo = 0.0      # segundos de ruta, para la cadencia real
        self.z_log = []         # z robusto máximo por ciclo, para el informe
        self.disp_log = []      # disparidad BRUTA máxima por ciclo
        # EPISODIOS: un episodio es "salió de la ruta y volvió". Es el
        # número que de verdad se quiere saber —con un trípode debería ser
        # 1— y no se puede deducir del contador de ciclos, porque el
        # limitador de pendiente estira cada episodio a ~133 ciclos.
        self.n_episodios = 0
        self.en_episodio = False
        self.pico_episodio = 0.0
        self.picos = []

    def preparar(self):
        if not self.activa:
            return False
        if self.vision is None:
            print('  [evasión] No hay cámara en este vuelo. Desactivada.')
            self.activa = False
            return False
        try:
            import evasion as ev
        except Exception as e:
            print(f'  [evasión] No se pudo importar evasion.py: {e}')
            self.activa = False
            return False
        self._ev = ev
        self.mapa = ev.MapaObstaculos(vida_media=EVASION_VIDA_MEDIA)
        if VISION_MODELO == 'depth_anything_v2_metrico':
            # Da metros y el estimador los entrega como 1/m: a=1, b=0, sin calibrar.
            self.cal = ev.CalibracionMiDaS(1.0, 0.0)
        else:
            self.cal = ev.CalibracionMiDaS(EVASION_CAL_A, EVASION_CAL_B)
            if VISION_MODELO != EVASION_CAL_MODELO:
                self.cal = ev.CalibracionMiDaS()
                print('  [evasión] Modelo distinto: no se reutiliza calibración métrica.')
        self.rumbos = ev.rumbos_columnas(EVASION_N_COL, EVASION_FOV_DEG)

        print()
        print('─' * 62)
        print(f'  EVASIÓN: modo {"ESQUIVAR" if self.esquiva else "OBSERVAR"}')
        print('─' * 62)
        if not self.esquiva:
            print('  OBSERVAR: detecta y anota, pero NO toca la ruta.')
            print('  Cuando el informe te convenza, pon EVASION_MODO =')
            print("  'esquivar'.")
        elif EVASION_DISP_MIN is None:
            # La lección de los vuelos del 16-sep: 'esquivar' sin compuerta
            # mueve la zanahoria 40 cm durante el 75% del vuelo por
            # fantasmas, y el seguimiento deja de parecerse a la ruta.
            # No se bloquea el vuelo —es tu dron— pero no se calla.
            print()
            print('  ' + '!' * 58)
            print('  ⚠ ESQUIVAR SIN COMPUERTA (EVASION_DISP_MIN = None)')
            print('  ' + '!' * 58)
            print('  Medido en vuelo sobre laboratorio VACÍO: el detector')
            print('  dispara en ~75% de los ciclos y manda la zanahoria al')
            print('  tope de 40 cm por obstáculos que no existen.')
            print()
            print('  CON ESTA CONFIGURACIÓN EL SEGUIMIENTO DE RUTA SE VA A')
            print('  DEGRADAR de ~4 cm de RMS a 20-35 cm. No es el control:')
            print('  es la evasión apartando el dron de fantasmas.')
            print()
            print('  Si lo que quieres es medir la RUTA, pon EVASION_MODO')
            print("  = 'off'. Si quieres depurar la evasión sin arriesgar")
            print("  la trayectoria, pon 'observar'.")
            print('  ' + '!' * 58)
        if not self.cal.valida:
            print('  Sin calibrar (EVASION_CAL_A/B = None): se usa solo la')
            print('  señal RELATIVA — sabe que hay algo cerca y en qué')
            print('  dirección, pero no a cuántos metros. Para calibrar:')
            print('    python volar_ruta.py --calibrar-midas')
        else:
            print(f'  Calibrada: a={self.cal.a:.5f}  b={self.cal.b:.4f}')

        print()
        print('  Filtros contra el laboratorio:')
        print(f'    franja de imagen  : {EVASION_FRANJA[0]:.2f}–'
              f'{EVASION_FRANJA[1]:.2f} del alto (fuera el suelo del encuadre)')
        print(f'    verticalidad      : ratio < {EVASION_MAX_RATIO} '
              f'(suelo ≈ 10, trípode ≈ 1)')
        if EVASION_DIST_RUTA:
            print(f'    zona              : solo a menos de '
                  f'{EVASION_DIST_RUTA*100:.0f} cm de la ruta')
        if EVASION_ZONAS_PROHIBIDAS:
            print(f'    zonas excluidas   : {len(EVASION_ZONAS_PROHIBIDAS)}')

        # EL AVISO IMPORTANTE
        if SEGUIR_YAW_RUTA and YAW_MODO == 'perpendicular':
            borde = YAW_ANGULO_DEG - EVASION_FOV_DEG / 2.0
            print()
            print('  NOTA sobre el encuadre (SOBRE EL PAPEL — comprueba el')
            print('  vídeo, que manda lo que se ve, no este cálculo):')
            print('  nominalmente la cámara iría a '
                  f'{YAW_ANGULO_DEG:.0f}° del avance.')
            print(f'     Con FOV de {EVASION_FOV_DEG:.0f}°, el encuadre cubre '
                  f'de {YAW_ANGULO_DEG-EVASION_FOV_DEG/2:.0f}° a '
                  f'{YAW_ANGULO_DEG+EVASION_FOV_DEG/2:.0f}°.')
            print(f'     Con eso el frente quedaría {borde:.0f}° fuera y no')
            print('     habría protección frontal.')
            print('     PERO ese cálculo depende de YAW_MONTAJE_DEG, que no')
            print('     está medido. Si en el vídeo ves que la cámara mira')
            print('     hacia adelante, entonces el montaje real compensa y')
            print('     esto no aplica: fíate del vídeo.')
            print('     Para dejarlo cerrado: --calibrar')
        print('─' * 62)
        return True

    def calibrar_fondo(self, segundos=None, mantener=None):
        """
        Mide el fondo de disparidad del laboratorio y fija la compuerta.

        Se llama con el dron QUIETO y sin nada delante. Devuelve el valor
        fijado, o None si no se pudo medir.

        POR QUÉ HACE FALTA UNA COMPUERTA SOBRE LA DISPARIDAD BRUTA
        En cualquier cuadro hay alguna columna por encima de 1.25 veces la
        mediana, aunque mires a una pared lisa: basta el ruido de MiDaS.
        Y el umbral de sigmas tampoco basta, porque en el laboratorio real
        hay objetos a distintas distancias que destacan DE VERDAD. El fallo
        no está en detectarlos, está en a qué distancia se colocan: la
        calibración a metros satura en 1.79 m, así que una pared a 3 m
        acaba anotada a 0.4 m, encima de la ruta. Por eso esta compuerta
        corta ANTES de convertir a metros.
        """
        if not self.activa or self.vision is None or self._ev is None:
            return None
        seg = EVASION_DISP_AUTO_SEG if segundos is None else float(segundos)
        muestras, visto, t0 = [], None, time.time()
        while time.time() - t0 < seg:
            # `mantener()` reenvía la consigna de posición cada vuelta. Sin
            # eso el dron pasa dos segundos sin recibir nada y se va a la
            # deriva justo antes de arrancar la ruta: medir el fondo no
            # puede costar la posición inicial.
            if mantener is not None and not mantener():
                return None
            u = self.vision.ultimo()
            mp, ts = u.get('mapa'), u.get('mapa_ts', 0.0)
            if mp is not None and ts != visto:
                visto = ts
                try:
                    disp, _ = self._ev.detectar_columnas_vertical(
                        mp, n_col=EVASION_N_COL, franja=EVASION_FRANJA,
                        percentil=EVASION_PERCENTIL)
                    muestras.append(float(np.max(disp)))
                except Exception:
                    pass
            time.sleep(0.02)

        if len(muestras) < EVASION_DISP_AUTO_MIN_N:
            self._fondo = list(muestras)      # no se tiran: valen igual
            fps = len(muestras) / max(seg, 1e-3)
            print(f'  [evasión] solo {len(muestras)} cuadros en {seg:.1f} s '
                  f'({fps:.1f} fps).')
            if EVASION_DISP_AUTO_EN_VUELO:
                faltan = EVASION_DISP_AUTO_MIN_N - len(muestras)
                print(f'            Se sigue midiendo durante la ruta; '
                      f'faltan {faltan} cuadros.')
                print(f'            Hasta entonces NO se mete nada en el '
                      f'mapa, así que el principio de la ruta tiene que '
                      f'estar despejado.')
            else:
                print('            La compuerta se queda sin poner.')
            return None
        p99 = float(np.percentile(muestras, 99))
        self.disp_fondo = p99 * EVASION_DISP_AUTO_MARGEN
        print(f'  [evasión] fondo medido con {len(muestras)} cuadros: '
              f'p50={np.percentile(muestras,50):.1f} p99={p99:.1f} → '
              f'compuerta en {self.disp_fondo:.1f}')
        print(f'            (para fijarla a mano: '
              f'EVASION_DISP_MIN = {self.disp_fondo:.1f})')
        return self.disp_fondo

    @property
    def compuerta(self):
        """El valor a mano manda; si no hay, el autocalibrado; si no, nada."""
        return EVASION_DISP_MIN if (EVASION_DISP_MIN is not None
                                    and VISION_MODELO == EVASION_CAL_MODELO) \
            else self.disp_fondo

    def _soltar_lado(self):
        """Deja de comprometerse con un lado: la maniobra terminó."""
        if self.mantener_compromiso:
            return
        self.lado_comp = 0
        self.t_comp = None
        self._ciclos_libre = 0

    def _suavizar(self, desvio, t_ahora):
        """
        Limita a cuánto puede MOVERSE el desvío por segundo.

        Tiene que aplicarse en TODAS las salidas de paso(), incluidas las
        de "no hay obstáculos": el mapa olvida con vida media de 2 s, así
        que se vacía a cada rato, y una salida sin limitar devolvería la
        zanahoria de golpe a la ruta. El tirón de vuelta es igual de malo
        que el de ida.
        """
        if self.t_desvio is None:
            dt = 0.02
        else:
            dt = max(min(t_ahora - self.t_desvio, 0.5), 1e-3)
        self.t_desvio = t_ahora
        objetivo = float(desvio)
        delta = objetivo - self.desvio_suave

        # ¿Este paso ALEJA de la ruta o ACERCA? Alejarse es |desvío|
        # creciendo, y ahí el retraso cuesta un choque, así que va rápido.
        # Volver solo cuesta un tirón, y va suave. Un cambio de signo
        # (el obstáculo cruza, o pasa a mandar otro) se trata como
        # regreso mientras cruza el cero, que es lo prudente: no se
        # dispara de +40 a −40 a toda velocidad.
        alejandose = abs(objetivo) > abs(self.desvio_suave)
        rate = EVASION_DESVIO_RATE_SAL if alejandose else EVASION_DESVIO_RATE
        paso_max = rate * dt

        if abs(delta) > paso_max:
            self.n_saturado += 1
            delta = paso_max if delta > 0 else -paso_max
        self.desvio_suave += delta
        return self.desvio_suave

    # Un desvío por debajo de esto no es una maniobra, es la cola del
    # limitador volviendo a cero. 2 cm: por debajo, el propio error de
    # seguimiento (p95 ≈ 10 cm) ya es mucho mayor.
    UMBRAL_EPISODIO = 0.02

    def _contar_episodio(self, desvio):
        """
        Cuenta MANIOBRAS, no ciclos.

        El contador de ciclos no sirve para juzgar la evasión: con el
        limitador de pendiente, un único fantasma que llegue al tope
        ocupa ~133 ciclos (subir 1.33 s + bajar 1.33 s), así que 1247
        ciclos pueden ser 9 episodios o 300. Lo que se quiere saber es
        cuántas VECES se apartó: con un trípode en la ruta, lo correcto
        es 1.

        Histéresis para no contar dos veces la misma maniobra si el
        desvío tiembla alrededor del umbral: entra en UMBRAL_EPISODIO y
        no sale hasta bajar de la mitad.
        """
        a = abs(float(desvio))
        if not self.en_episodio:
            if a >= self.UMBRAL_EPISODIO:
                self.en_episodio = True
                self.n_episodios += 1
                self.pico_episodio = a
        else:
            self.pico_episodio = max(self.pico_episodio, a)
            if a < self.UMBRAL_EPISODIO * 0.5:
                self.en_episodio = False
                self.picos.append(self.pico_episodio)
                self.pico_episodio = 0.0

    def paso(self, pos_xy, rumbo_camara, p_carrot, tangente, t_ahora):
        """
        Devuelve (desvio_lateral_m, parar, dist_min).
        En modo 'observar' devuelve siempre desvio=0 y parar=False, pero
        registra lo que habría hecho.
        """
        if not self.activa:
            return 0.0, False, np.inf

        u = self.vision.ultimo()
        mp = u.get('mapa')
        mp_dist = u.get('mapa_dist')
        ts = u.get('mapa_ts', 0.0)
        if mp is not None and ts != self.ts_visto and rumbo_camara is not None:
            self.ts_visto = ts
            self.vista = {'ts': ts, 'valida': False, 'bloqueada': True}
            try:
                if not np.all(np.isfinite(mp)):
                    raise ValueError('Mapa de profundidad no finito')
                disp, ratio = self._ev.detectar_columnas_vertical(
                    mp, n_col=EVASION_N_COL, franja=EVASION_FRANJA,
                    percentil=EVASION_PERCENTIL)
                if mp_dist is not None and (np.shape(mp_dist) != mp.shape
                                            or not np.all(np.isfinite(mp_dist))):
                    mp_dist = None
                val_dist = (self._ev.valor_por_columna(
                    mp, mp_dist, disp, n_col=EVASION_N_COL, franja=EVASION_FRANJA)
                    if mp_dist is not None else disp)
                # Se registra SIEMPRE, dispare o no: es lo que permite
                # elegir EVASION_SIGMAS con datos de vuelo en vez de a ojo.
                self.z_log.append(float(np.max(self._ev.z_robusto(disp))))
                self.disp_log.append(float(np.max(disp)))

                # ── COMPUERTA: se sigue midiendo en vuelo si hace falta ──
                # Mientras no esté puesta no se alimenta el mapa: las
                # detecciones de este rato solo miden el fondo. Sin
                # compuerta no se puede separar un trípode del ruido, y
                # meterlas igual es exactamente cómo se fabrican fantasmas.
                if (self.compuerta is None and EVASION_DISP_AUTO
                        and EVASION_DISP_AUTO_EN_VUELO):
                    self._fondo.append(float(np.max(disp)))
                    self.n_sin_compuerta += 1
                    if len(self._fondo) >= EVASION_DISP_AUTO_MIN_N:
                        self.disp_fondo = float(
                            np.percentile(self._fondo, 99)
                            * EVASION_DISP_AUTO_MARGEN)
                        print(f'  [evasión] compuerta puesta en '
                              f'{self.disp_fondo:.0f} con '
                              f'{len(self._fondo)} cuadros de fondo '
                              f'(para fijarla: EVASION_DISP_MIN = '
                              f'{self.disp_fondo:.0f})')
                    raise _MidiendoFondo
                sosp = self._ev.columnas_sospechosas(
                    disp, EVASION_UMBRAL_REL,
                    ratio=ratio, max_ratio=EVASION_MAX_RATIO,
                    sigmas=EVASION_SIGMAS, disp_min=self.compuerta)
                # Vista sin filtro por zona: también protege fuera de la ruta.
                self.vista = {'ts': ts, 'valida': self.compuerta is not None,
                              'bloqueada': bool(np.any(sosp))}
                self.n_suelo += int(np.sum(
                    (disp > EVASION_UMBRAL_REL * np.median(disp))
                    & (ratio >= EVASION_MAX_RATIO)))
                if np.any(sosp):
                    if self.cal.valida:
                        d = self.cal.metros(val_dist[sosp])
                    else:
                        # Sin calibrar: se coloca a una distancia fija
                        # prudente. Sirve para la dirección, no para la
                        # distancia — y por eso 'observar' va primero.
                        d = np.full(int(sosp.sum()), 0.8)
                    ok = np.isfinite(d) & (d > 0.15) & (d < EVASION_DIST_MAX)
                    if np.any(ok):
                        pts = self._ev.a_mundo(pos_xy, rumbo_camara,
                                               self.rumbos[sosp][ok], d[ok])
                        # Filtro por ZONA: quita la estructura fija del
                        # laboratorio, que es un objeto real y vertical y
                        # por tanto sobrevive al test de verticalidad.
                        dentro = self._ev.filtrar_por_zona(
                            pts, ruta_xy=self.ruta_xy,
                            dist_max=EVASION_DIST_RUTA,
                            zonas_prohibidas=EVASION_ZONAS_PROHIBIDAS)
                        self.n_fuera += int(np.sum(~dentro))
                        pts = pts[dentro]
                        if len(pts):
                            self.mapa.actualizar(pts, t_ahora, peso=1.0)
                            self.n_det += len(pts)
            except _MidiendoFondo:
                pass          # no es un fallo: es el rato de calibración
            except Exception as e:
                # NADA DE `pass`. Tragarse esto era el motivo de que no se
                # pudiera distinguir un detector que funciona de uno
                # muerto: si MiDaS reventaba, si la calibración no valía o
                # si la proyección fallaba, el mapa dejaba de actualizarse
                # en silencio y los obstáculos viejos se quedaban mandando.
                # Se cuenta, se guarda el primero y, si falla siempre, se
                # apaga la evasión y se dice.
                self.n_errores += 1
                if self.primer_error is None:
                    self.primer_error = f'{type(e).__name__}: {e}'
                    print(f'  ⚠ [evasión] fallo procesando la cámara: '
                          f'{self.primer_error}')
                if self.n_errores >= EVASION_ERRORES_MAX:
                    self.activa = False
                    print(f'  ✗ [evasión] {self.n_errores} fallos seguidos '
                          f'procesando la cámara. EVASIÓN DESACTIVADA para '
                          f'el resto del vuelo; se sigue la ruta nominal.')
                    return self._suavizar(0.0, t_ahora), False, np.inf
            else:
                self.n_errores = 0

        # OJO: estas dos salidas TAMBIÉN pasan por el limitador. Es el caso
        # más frecuente de todos —el mapa olvida los obstáculos con vida
        # media de 2 s, así que se vacía continuamente— y si devolvieran un
        # 0.0 duro, la zanahoria volvería de 40 cm a 0 en un ciclo. Sería
        # el mismo tirón que el limitador viene a quitar, solo que de
        # vuelta. Se vuelve a la ruta con la misma suavidad con que se sale.
        if self.mapa is None:
            return self._suavizar(0.0, t_ahora), False, np.inf

        # ── EL OLVIDO, CADA CICLO ────────────────────────────────────
        # Antes el decaimiento vivía dentro de `mapa.actualizar()`, y a
        # `actualizar()` solo se le llamaba cuando llegaban detecciones
        # nuevas que habían pasado todos los filtros. O sea que si la
        # cámara callaba, el objeto salía del encuadre o MiDaS fallaba en
        # silencio, un obstáculo confirmado se quedaba con su peso intacto
        # PARA SIEMPRE.
        #
        # Eso es lo que produjo el vuelo del 16-sep: zanahoria al tope de
        # 40 cm durante 1215 de 1620 ciclos —el 75 % del vuelo— empujada
        # por un fantasma que se había congelado a 1.12 m del trípode real.
        self.mapa.decaer(t_ahora)

        obst, pesos = self.mapa.confirmados(EVASION_PESO_MIN)
        if len(obst) == 0:
            self._soltar_lado()
            return self._suavizar(0.0, t_ahora), False, np.inf

        # ══════════════════════════════════════════════════════════
        # COMPROMISO DE LADO  (17-sep-2026)
        # ══════════════════════════════════════════════════════════
        # El fallo: «esquiva el objeto, pero nunca se hace para adelante,
        # se va de izquierda a derecha y lo intenta esquivar otra vez».
        #
        # `desvio_lateral` elegía lado en CADA ciclo, con los obstáculos de
        # ESE ciclo. Un trípode no da un punto fijo: da un racimo que
        # cambia en cada cuadro, porque se ve desde otro ángulo y entran y
        # salen puntos del mapa. Con el racimo cambian `max(l)` y `min(l)`,
        # y en cuanto «lo que cuesta ir por la izquierda» y «lo que cuesta
        # ir por la derecha» se cruzan, el lado elegido salta.
        #
        # Y saltaba a 50 Hz. El dron salía hacia un lado, el cálculo
        # cambiaba de idea, salía hacia el otro, y así: eses delante del
        # trípode sin pasarlo nunca. Rodear algo requiere MANTENER una
        # decisión mientras se ejecuta, y eso es memoria, no geometría.
        #
        # Ahora: en cuanto el desvío pasa de EVASION_COMPROMISO_MIN se fija
        # el lado, y no se suelta hasta que el obstáculo deja de estar
        # delante (o hasta EVASION_COMPROMISO_MAX_S, por si algo se atasca).
        desvio, dmin, no_cabe = self._ev.desvio_lateral(
            p_carrot, tangente, obst, pesos,
            radio_influencia=EVASION_RADIO_INF,
            radio_seguro=EVASION_RADIO_SEG,
            desvio_max=EVASION_DESVIO_MAX,
            holgura=EVASION_HOLGURA,
            lado=self.lado_comp,
            margen_lado=EVASION_MARGEN_LADO,
            cola=EVASION_COLA)

        if self.lado_comp == 0:
            if abs(desvio) >= EVASION_COMPROMISO_MIN:
                self.lado_comp = 1 if desvio > 0 else -1
                self.t_comp = t_ahora
                self.n_compromisos += 1
        else:
            # Se suelta cuando ya no hay nada que esquivar por ese lado
            # (el obstáculo quedó atrás) o si se agota el tiempo.
            if abs(desvio) < 1e-6:
                self._ciclos_libre += 1
                if self._ciclos_libre >= EVASION_CICLOS_SOLTAR:
                    self._soltar_lado()
            else:
                self._ciclos_libre = 0
                if (not self.mantener_compromiso and self.t_comp is not None
                        and t_ahora - self.t_comp > EVASION_COMPROMISO_MAX_S):
                    self.n_comp_vencidos += 1
                    self._soltar_lado()

        if no_cabe:
            self.n_no_cabe += 1

        # ── LIMITADOR DE PENDIENTE ──────────────────────────────────
        # desvio_lateral() no tiene memoria: calcula el desvío del ciclo
        # a partir de los obstáculos de ESE ciclo. Como el bucle corre a
        # 50 Hz y la consigna es de POSICIÓN, un obstáculo que aparece
        # movía la zanahoria de 0 a 40 cm en 20 ms — un escalón de 20 m/s
        # en la consigna. Y si el signo cambiaba (el obstáculo cruza la
        # trayectoria, o pasa a mandar otro), el salto era de 80 cm.
        # De ahí el tirón que se siente en vuelo.
        #
        # Aquí el desvío se mueve como mucho a EVASION_DESVIO_RATE m/s.
        # No se suaviza con un filtro exponencial a propósito: un filtro
        # deja pasar parte del escalón inmediatamente, un limitador de
        # pendiente no deja pasar nada por encima del límite.
        #
        # MARGEN: el obstáculo empieza a contar a EVASION_RADIO_INF
        # (0.60 m). A la velocidad de crucero (0.25 m/s) eso son 2.4 s
        # hasta el contacto, y a 0.30 m/s el desvío llega al tope de
        # 40 cm en 1.33 s. Queda margen, pero si subes la velocidad de
        # vuelo hay que subir esto en la misma proporción.
        desvio = self._suavizar(desvio, t_ahora)

        parar, d_frente = self._ev.hay_que_parar(
            obst, pos_xy, tangente,
            radio_parada=EVASION_RADIO_PARO,
            medio_ancho=EVASION_ANCHO_PARO)
        # NO CABE = parar. Colarse por un hueco más estrecho que la holgura
        # es chocar, solo que despacio. `hay_que_parar` solo mira lo que hay
        # justo delante a menos de EVASION_RADIO_PARO (22 cm), y para
        # entonces ya es tarde: esto lo ve a 60 cm y con la geometría
        # completa de los dos lados.
        parar = parar or no_cabe

        if abs(desvio) > 1e-3:
            self.n_desvios += 1
            self.desvio_max_visto = max(self.desvio_max_visto, abs(desvio))
        self._contar_episodio(desvio)
        if parar:
            self.n_paradas += 1
        self.dist_min_vista = min(self.dist_min_vista, dmin)
        self.registro.append((t_ahora, desvio, dmin, len(obst)))

        if not self.esquiva:
            return 0.0, False, dmin          # observar: no toca nada
        return desvio, parar, dmin

    def informe(self):
        # Nada que contar solo si la evasión NUNCA estuvo activa (modo
        # 'off'). Si se desactivó EN VUELO por fallos de cámara, el informe
        # se imprime igual: es justo el vuelo en que más falta hacen los
        # histogramas de z y de disparidad para saber qué pasó.
        if not self.activa and not self.n_errores and not self.z_log:
            return
        print()
        if self.n_errores >= EVASION_ERRORES_MAX:
            print('  ── EVASIÓN: SE DESACTIVÓ DURANTE EL VUELO ──')
            print(f'    {self.n_errores} fallos seguidos procesando la cámara.')
            print(f'    El primero: {self.primer_error}')
            print('    A partir de ahí se voló la ruta nominal. Lo de abajo')
            print('    es lo que se alcanzó a medir antes de apagarse.')
        else:
            print('  ── EVASIÓN ──')
        obst, pesos = (self.mapa.confirmados(EVASION_PESO_MIN)
                       if self.mapa else (np.zeros((0, 2)), None))
        print(f'  Detecciones proyectadas : {self.n_det}')
        print(f'  Descartadas por SUELO   : {self.n_suelo} '
              f'(gradiente vertical alto)')
        print(f'  Descartadas por ZONA    : {self.n_fuera} '
              f'(lejos de la ruta o en zona excluida)')
        print(f'  Obstáculos confirmados  : {len(obst)} '
              f'(vistos ≥{EVASION_PESO_MIN:.0f} veces)')
        for p, w in list(zip(obst, pesos))[:8]:
            print(f'      ({p[0]:+.2f}, {p[1]:+.2f})  peso {w:.1f}')
        if np.isfinite(self.dist_min_vista):
            print(f'  Se acercó a             : '
                  f'{self.dist_min_vista*100:.0f} cm del más cercano')
        # LO PRIMERO, porque es el número que dice si esto vale o no:
        # con un trípode junto a la ruta, lo correcto es 1 maniobra.
        picos = list(self.picos) + ([self.pico_episodio]
                                    if self.en_episodio else [])
        v2 = 'Maniobras de evasión' if self.esquiva else 'SE HABRÍA apartado'
        print(f'  {v2:24}: {self.n_episodios} '
              f'{"vez" if self.n_episodios == 1 else "veces"}'
              f'  (≥{self.UMBRAL_EPISODIO*100:.0f} cm cada una)')
        if picos:
            print(f'      desvío de cada una    : '
                  + ', '.join(f'{p*100:.0f}' for p in picos[:10])
                  + (' …' if len(picos) > 10 else '') + ' cm')
        if self.n_episodios > 3:
            print('      ⚠ Más de 3 maniobras. Si no había tantos')
            print('        obstáculos, son fantasmas: NO actives '
                  "'esquivar'.")

        verbo = 'se apartó' if self.esquiva else 'SE HABRÍA apartado'
        print(f'  Ciclos en que {verbo}: {self.n_desvios}'
              f'  (máx {self.desvio_max_visto*100:.0f} cm)')
        print('      (ciclos del bucle a 50 Hz, no maniobras: el limitador')
        print('       estira cada maniobra a ~133 ciclos)')
        if self.n_saturado:
            print(f'  Desvío limitado en       : {self.n_saturado} ciclos '
                  f'(tope {EVASION_DESVIO_RATE:.2f} m/s)')
        if self.n_paradas:
            v2 = 'paró' if self.esquiva else 'SE HABRÍA parado'
            print(f'  Ciclos en que {v2}     : {self.n_paradas}')
        if self.n_compromisos:
            print(f'  Lado fijado              : {self.n_compromisos} '
                  f'{"vez" if self.n_compromisos == 1 else "veces"}')
            print('      Una por maniobra. Si sale MUCHO más que el número de')
            print('      maniobras de arriba, el dron estuvo dudando de lado:')
            print('      sube EVASION_COMPROMISO_MIN o EVASION_CICLOS_SOLTAR.')
        if self.n_comp_vencidos:
            print(f'  Compromisos vencidos     : {self.n_comp_vencidos} '
                  f'(pasaron de {EVASION_COMPROMISO_MAX_S:.0f} s)')
            print('      Se quedó rodeando algo más tiempo del previsto.')
        print(f'  Holgura pedida           : {EVASION_HOLGURA*100:.0f} cm '
              f'del punto detectado al centro del dron')
        print(f'      (separación con cinta ≈ 0.65× → unos '
              f'{EVASION_HOLGURA*65:.0f} cm)')
        print(f'      (tope de desvío {EVASION_DESVIO_MAX*100:.0f} cm; tiene '
              'que ser mayor que la holgura)')
        if self.n_no_cabe:
            print(f'  HUECO DEMASIADO ESTRECHO : {self.n_no_cabe} ciclos')
            print('      No había ningún desvío que dejara la holgura pedida')
            print('      (obstáculos a los dos lados, o uno muy ancho), así')
            print('      que se trató como PARADA en vez de colarse.')
            print('      Si esto sale mucho y el laboratorio está despejado,')
            print('      sobran fantasmas o EVASION_HOLGURA es muy grande.')
        # ── LO QUE SIRVE PARA ELEGIR EVASION_SIGMAS ──
        # El z es adimensional y no depende de la calibración ni de la
        # escena, así que este histograma SÍ se puede comparar entre
        # vuelos y entre laboratorios.
        if self.n_sin_compuerta:
            print(f'  Cuadros usados para medir el fondo: '
                  f'{self.n_sin_compuerta} (no entraron en el mapa)')

        # ── LA CADENCIA DE LA CÁMARA, QUE ES EL CUELLO DE BOTELLA ──
        # Con el AI-deck en RAW el enlace se satura y bajan ~0.8 cuadros
        # por segundo. A 0.10 m/s el dron recorre 13 cm entre cuadro y
        # cuadro: no hay evasión posible con eso, por muy bien que esté
        # el resto. El propio [cam] lo dice en cada línea del vuelo.
        if self.z_log and self.t_vuelo and self.t_vuelo > 1.0:
            fps = len(self.z_log) / self.t_vuelo
            print()
            print(f'  ── CADENCIA DE LA CÁMARA ──')
            print(f'  {len(self.z_log)} cuadros en {self.t_vuelo:.0f} s = '
                  f'{fps:.2f} fps')
            if fps < 3.0:
                salto = 0.10 / max(fps, 1e-3)
                print(f'  ⚠ DEMASIADO LENTA. El dron recorre {salto*100:.0f} cm')
                print( '    entre cuadro y cuadro. La evasión no puede')
                print( '    funcionar así, independientemente de los umbrales.')
                print( '    ARREGLO: pasar el GAP8 a JPEG. RAW satura el')
                print( '    enlace y es lo que avisa [cam] en cada línea.')

        if self.z_log:
            z = np.asarray(self.z_log, float)
            print()
            print(f'  ── CUÁNTO DESTACA LO MÁS CERCANO (z robusto) ──')
            print(f'  {len(z)} cuadros | umbral actual EVASION_SIGMAS = '
                  f'{EVASION_SIGMAS}')
            ps = [50, 90, 95, 99]
            vs = np.percentile(z, ps)
            print('      ' + '  '.join(f'p{p}={v:.2f}' for p, v in zip(ps, vs))
                  + f'  máx={z.max():.2f}')
            if EVASION_SIGMAS is not None:
                frac = 100.0 * float(np.mean(z > EVASION_SIGMAS))
                print(f'  Cuadros por encima del umbral: {frac:.1f} %')
                print()
                if frac > 20.0:
                    print('  ⚠ Dispara en demasiados cuadros. Si en este vuelo')
                    print('    NO había obstáculos, sube EVASION_SIGMAS por')
                    print(f'    encima del p99 ({vs[-1]:.2f}).')
                elif frac < 1.0:
                    print('  Dispara poco. Si había un obstáculo y no lo vio,')
                    print(f'    baja EVASION_SIGMAS hacia el p90 ({vs[1]:.2f}).')

        # ── LO QUE SIRVE PARA ELEGIR EVASION_DISP_MIN ──
        # Disparidad BRUTA, sin pasar por la calibración. Es el número con
        # el que se separa "cerca" de "lejos" sin depender de unos metros
        # que no son de fiar.
        if self.disp_log:
            dd = np.asarray(self.disp_log, float)
            ps = [50, 90, 95, 99]
            vs = np.percentile(dd, ps)
            print()
            print('  ── DISPARIDAD BRUTA DE LO MÁS CERCANO ──')
            print('      ' + '  '.join(f'p{p}={v:.0f}' for p, v in zip(ps, vs))
                  + f'  máx={dd.max():.0f}')
            g = self.compuerta
            if g is None:
                print('  SIN COMPUERTA (EVASION_DISP_MIN=None y la')
                print('  autocalibración no midió nada).')
                print('  Si este vuelo fue SIN obstáculos, ese es el fondo de')
                print(f'  tu laboratorio: pon EVASION_DISP_MIN = {vs[-1]:.0f}')
                print('  (el p99) y vuelve a volar, ahora con los trípodes.')
            else:
                de_donde = ('a mano' if EVASION_DISP_MIN is not None
                            else 'autocalibrada al despegar')
                frac = 100.0 * float(np.mean(dd > g))
                print(f'  Compuerta {g:.0f} ({de_donde}) → la supera el '
                      f'{frac:.1f} % de los cuadros.')
                if frac > 30.0:
                    print('  ⚠ La supera demasiado. Si en este vuelo no había')
                    print('    obstáculos, son fantasmas: súbela por encima')
                    print(f'    del p99 ({vs[-1]:.0f}).')
                elif frac < 0.5 and self.n_episodios == 0:
                    print('  No la supera casi nunca. Si había un trípode y no')
                    print(f'    lo vio, bájala hacia el p90 ({vs[1]:.0f}).')

        if not self.esquiva:
            print()
            print('  Modo OBSERVAR: no se tocó la ruta. Mira si los')
            print('  obstáculos confirmados coinciden con lo que había de')
            print('  verdad en el laboratorio. Si sí, EVASION_MODO =')
            print("  'esquivar'. Si salen fantasmas, sube EVASION_UMBRAL_REL")
            print(f'  ({EVASION_UMBRAL_REL}) o EVASION_PESO_MIN '
                  f'({EVASION_PESO_MIN:.0f}).')


def calibrar_midas():
    """
    Ajusta disparidad de MiDaS → metros.  --calibrar-midas

    MiDaS da profundidad inversa relativa, no metros, y encima con una
    pérdida invariante a escala. Un ajuste global funciona de forma
    aproximada; esto lo mide en vez de suponerlo.

    Modelo:  distancia = 1 / (a·disparidad + b)
    Es el correcto porque disparidad ∝ 1/distancia. Probado contra un
    ajuste lineal: 4 cm de error medio contra 58 cm.

    MUESTREA IGUAL QUE EL DETECTOR, Y ESO NO ES UN DETALLE
    Antes esto tomaba un recuadro central fijo (35-65% del alto) mientras
    que en vuelo `Evasion.paso` usa `detectar_columnas_vertical`, que mira
    la franja EVASION_FRANJA y se queda con su mitad SUPERIOR. Dos
    consecuencias, las dos malas:

      1. El recuadro bajaba hasta el 65% del alto, donde entra el SUELO.
         Con el dron apoyado, el suelo está a centímetros del lente, y el
         percentil 90 se queda con el 10% de píxeles más CERCANOS: ganaba
         siempre el suelo. La disparidad se quedaba congelada en ~300-340
         sin responder al objeto.
      2. Aun sin suelo, a y b ajustados sobre una ventana NO valen para
         los números que produce otra ventana. La calibración medía una
         cosa y el detector consumía otra.

    Ahora llama a la MISMA función con los MISMOS parámetros y toma la
    columna central, que es donde pones el objeto. Lo que se ajusta es
    exactamente lo que después se convierte a metros en vuelo.
    """
    print()
    print('═' * 62)
    print('  CALIBRACIÓN DE PROFUNDIDAD (MiDaS → metros)')
    print('═' * 62)
    print('  Necesitas el dron encendido con la cámara, y una cinta')
    print('  métrica. Se mide la COLUMNA CENTRAL de la imagen, en la')
    print(f'  franja {EVASION_FRANJA[0]:.2f}-{EVASION_FRANJA[1]:.2f} del alto '
          '(mitad superior): lo mismo')
    print('  que mira la evasión en vuelo.')
    if VISION_MODELO == 'hibrido':
        print('  Se calibra la distancia de MiDaS en los píxeles que marca Depth Anything.')
    print()
    print('  1. Cámara a la ALTURA DE VUELO y horizontal, para que el')
    print('     suelo quede por debajo del encuadre útil.')
    print('  2. El objeto, CENTRADO y llenando la columna del medio.')
    print('  3. Fondo quieto: mueve solo el objeto entre medida y medida.')
    print('  4. Mínimo 3 puntos, 5-6 mejor, repartidos de 0.4 a 2 m.')
    print('═' * 62)

    vis = P.SistemaVision()
    if not vis.iniciar():
        print('  ⚠ No se pudo iniciar la cámara.')
        return
    print('  Esperando a que MiDaS cargue...')
    t0 = time.time()
    while not vis.midas_listo and time.time() - t0 < 90:
        time.sleep(0.5)
    if not vis.midas_listo:
        print('  ⚠ MiDaS no cargó.')
        vis.detener()
        return

    import evasion as ev
    i_centro = EVASION_N_COL // 2
    disps, dists, ratios = [], [], []
    print(f'  Midiendo la columna {i_centro + 1} de {EVASION_N_COL} '
          '(la central).')
    try:
        while True:
            s = input('\n  Distancia real en metros (ENTER para terminar): ')
            s = s.strip().replace(',', '.')
            if not s:
                break
            try:
                z = float(s)
            except ValueError:
                print('  Número no válido.')
                continue
            if not 0.1 <= z <= 6.0:
                print('  Fuera de rango razonable (0.1 - 6 m).')
                continue
            muestras, muestras_r = [], []
            print('  Midiendo 3 s, no muevas nada...')
            t0 = time.time()
            visto = None
            while time.time() - t0 < 3.0:
                u = vis.ultimo()
                mp, ts = u.get('mapa'), u.get('mapa_ts', 0.0)
                mp_dist = u.get('mapa_dist')
                if mp is not None and ts != visto:
                    visto = ts
                    # EXACTAMENTE el mismo muestreo que Evasion.paso en
                    # vuelo: misma función, misma franja, mismo percentil.
                    sup, rat = ev.detectar_columnas_vertical(
                        mp, n_col=EVASION_N_COL, franja=EVASION_FRANJA,
                        percentil=EVASION_PERCENTIL)
                    if mp_dist is not None and (np.shape(mp_dist) != mp.shape
                                                or not np.all(np.isfinite(mp_dist))):
                        mp_dist = None
                    val_dist = (ev.valor_por_columna(
                        mp, mp_dist, sup, n_col=EVASION_N_COL, franja=EVASION_FRANJA)
                        if mp_dist is not None else sup)
                    muestras.append(float(val_dist[i_centro]))
                    muestras_r.append(float(rat[i_centro]))
                time.sleep(0.05)
            if len(muestras) < 2:
                print('  ⚠ No llegaron frames. ¿La cámara está viva?')
                continue
            d = float(np.median(muestras))
            rr = float(np.median(muestras_r))
            sd = float(np.std(muestras))
            disps.append(d)
            dists.append(z)
            ratios.append(rr)
            print(f'  ✓ {z:.2f} m → disparidad {d:.1f} '
                  f'({len(muestras)} frames, dispersión {sd:.1f}, '
                  f'ratio {rr:.2f})')
            if VISION_MODELO == 'depth_anything_v2_metrico':
                print(f'    modelo métrico: estima {1.0 / max(d, 1e-6):.2f} m '
                      f'(real {z:.2f} m)')
            if rr >= EVASION_MAX_RATIO:
                print(f'    ⚠ ratio {rr:.2f} ≥ {EVASION_MAX_RATIO}: la mitad '
                      'de abajo está mucho más cerca que la')
                print('      de arriba, o sea que esto parece SUELO y no un '
                      'objeto vertical.')
                print('      Sube la cámara o apúntala más horizontal, y '
                      'repite este punto.')
            if sd > 0.15 * max(d, 1e-6):
                print(f'    ⚠ dispersión {100.0 * sd / max(d, 1e-6):.0f}% del '
                      'valor: MiDaS no está estable aquí.')
                print('      Repite el punto; si sigue, el objeto no destaca '
                      'del fondo.')
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        vis.detener()

    formato_cal = _formato_camara(vis)
    if VISION_MODELO == 'depth_anything_v2_metrico':
        if disps:
            est = 1.0 / np.maximum(np.asarray(disps, float), 1e-6)
            err = np.abs(est - np.asarray(dists, float))
            print(f'\n  Modelo métrico: error medio {err.mean() * 100:.1f} cm | '
                  f'máx {err.max() * 100:.1f} cm en {len(disps)} medidas.')
        print('  No hay nada que ajustar ni copiar: este modelo ya da metros.')
        return

    if len(disps) < 3:
        print('\n  Hacen falta al menos 3 medidas. No se ajustó nada.')
        return

    # ── Antes de ajustar: ¿los datos pueden venir de una cámara? ──────
    # Un ajuste siempre sale. Que salga no quiere decir que sirva.
    orden = np.argsort(dists)
    z_ord = np.asarray(dists, float)[orden]
    d_ord = np.asarray(disps, float)[orden]
    sano = True

    if np.any(np.diff(d_ord) > 0.0):
        sano = False
        print('\n  ⚠ LA DISPARIDAD NO BAJA AL ALEJARTE:')
        for z, d in zip(z_ord, d_ord):
            print(f'        {z:5.2f} m  →  {d:7.1f}')
        print('    La disparidad va como 1/distancia, así que tiene que')
        print('    bajar SIEMPRE. Si sube, el percentil no está midiendo tu')
        print('    objeto: o se cuela el suelo, o el objeto no llena la')
        print('    columna central y está midiendo el fondo.')

    if z_ord[-1] / max(z_ord[0], 1e-6) < 2.5:
        sano = False
        print(f'\n  ⚠ Rango corto ({z_ord[0]:.2f}-{z_ord[-1]:.2f} m). Con los')
        print('    puntos apiñados el ajuste extrapola mal fuera de ellos.')
        print('    Cubre de 0.4 a 2 m, que es donde vas a volar.')

    n_suelo = int(np.sum(np.asarray(ratios) >= EVASION_MAX_RATIO))
    if n_suelo:
        sano = False
        print(f'\n  ⚠ {n_suelo} de {len(ratios)} medidas tienen pinta de '
              'SUELO (ratio alto).')

    cal = ev.CalibracionMiDaS()
    try:
        r = cal.ajustar(disps, dists)
    except Exception as e:
        print(f'\n  ⚠ No se pudo ajustar: {e}')
        return

    print()
    print('─' * 62)
    print('  distancia real → estimada por el ajuste')
    for z, d in zip(dists, disps):
        print(f'      {z:5.2f} m  →  {float(cal.metros(d)):5.2f} m')
    print('─' * 62)
    print(f'  Error medio {r["err_medio"]*100:.1f} cm | '
          f'máx {r["err_max"]*100:.1f} cm | R² {r["r2"]:.4f}')

    if r['err_medio'] > 0.20:
        sano = False
        print()
        print('  ⚠ Error medio alto. MiDaS es invariante a escala y su')
        print('    salida cambia con la escena: repítelo en un sitio')
        print('    parecido a donde vas a volar.')

    # El modelo 1/(a·d+b) explota donde a·d+b = 0. Con b<0 eso cae en una
    # disparidad POSITIVA, o sea alcanzable: por debajo da metros negativos.
    if r['b'] < 0.0 and r['a'] > 1e-12:
        d_sing = -r['b'] / r['a']
        sano = False
        print()
        print(f'  ⚠ El modelo se va a infinito en disparidad {d_sing:.0f} y')
        print('    da distancias NEGATIVAS por debajo de ahí. El fondo de la')
        print('    sala baja de eso en vuelo, así que estas constantes')
        print('    producirían basura.')

    print()
    if sano:
        print('  Copia esto en volar_ruta.py:')
        print()
        print(f'      EVASION_CAL_A = {r["a"]:.6f}')
        print(f'      EVASION_CAL_B = {r["b"]:.6f}')
        print(f'      EVASION_CAL_FORMATO = {formato_cal!r}')
    else:
        print('  ✗ NO COPIES ESTOS COEFICIENTES. Los avisos de arriba dicen')
        print('    que los datos no dan para un ajuste fiable:')
        print()
        print(f'        EVASION_CAL_A = {r["a"]:.6f}   ← solo informativo')
        print(f'        EVASION_CAL_B = {r["b"]:.6f}   ← solo informativo')
        print(f'        EVASION_CAL_FORMATO = {formato_cal!r}   ← solo informativo')
        print()
        print('    Con EVASION_CAL_A/B = None la evasión sigue funcionando')
        print('    con la señal relativa, que no depende de esto. Es mejor')
        print('    eso que una calibración mala.')
    print('─' * 62)


def calibrar_montaje():
    """
    Mide YAW_MONTAJE_DEG de verdad.   python volar_ruta.py --calibrar

    POR QUÉ HACE FALTA, Y POR QUÉ NO VALE MEDIRLO EN VUELO
    El lazo de yaw fuerza (y_moc − M) → dirección deseada, donde M es el
    montaje SUPUESTO. La nariz FÍSICA apunta a (y_moc − M_real), o sea:

        nariz física = dirección deseada + (M − M_real)

    Un error en M gira la cámara un ángulo constante… y NO aparece en
    ninguna medida que haga el propio lazo, porque M se cancela. El
    VerificadorCamara del vuelo anterior daba +0.0° incluso con la nariz
    a 90°: comprobaba que el lazo cerraba, no hacia dónde miraba la
    cámara. Para tu problema era inútil, y te lo vendí como la
    comprobación buena.

    La única forma de saber M_real es una observación FÍSICA: poner el
    dron mirando a una dirección conocida y leer lo que dice el MoCap.
    Es lo que hace esto. M es una propiedad de cómo definiste el cuerpo
    rígido en Motive, así que se mide una vez y ya.
    """
    import threading
    print()
    print('═' * 62)
    print('  CALIBRACIÓN DEL MONTAJE DE LA CÁMARA')
    print('═' * 62)
    print('  1. Dron EN EL SUELO, sin hélices girando.')
    print('  2. Apúntalo con la NARIZ (la cámara) hacia el eje +X del')
    print('     laboratorio: el mismo +X del eje horizontal de tus')
    print('     gráficas de ruta.')
    print('  3. Déjalo quieto y pulsa ENTER.')
    print('═' * 62)

    threading.Thread(target=P.start_mqtt, daemon=True).start()
    print('\n  Esperando al MoCap...')
    t0 = time.time()
    while P._mocap_ori is None:
        if time.time() - t0 > 15.0:
            print('  ⚠ El MoCap no manda orientación: no se puede calibrar.')
            print('    Node-RED tiene que incluir el cuaternión en el payload.')
            return
        time.sleep(0.2)
    print('  ✓ MoCap recibido.')

    try:
        input('\n  Nariz apuntando a +X y quieto → ENTER: ')
    except EOFError:
        pass

    vals = []
    for _ in range(40):
        y = P.mocap_pose.get('yaw', None)
        if y is not None:
            vals.append(float(y))
        time.sleep(0.05)
    if len(vals) < 10:
        print('  ⚠ No llegaron muestras suficientes.')
        return

    v = np.array(vals)
    m = float(np.arctan2(np.sin(v).mean(), np.cos(v).mean()))
    disp = float(np.degrees(np.std(_wrap(v - m))))
    # nariz_mundo = y_moc − M, y la nariz está en 0° → M = y_moc
    M = float(np.degrees(m)) % 360.0

    print()
    print('─' * 62)
    print(f'  Yaw del MoCap con la nariz en +X : {np.degrees(m):+.1f}°')
    print(f'  Dispersión                       : {disp:.2f}°')
    print('─' * 62)
    if disp > 3.0:
        print('  ⚠ Dispersión alta: ¿se movió el dron, o el cuerpo rígido')
        print('    está perdiendo marcadores? Repite la medida.')
    print()
    print(f'      YAW_MONTAJE_DEG = {M:.1f}')
    print()
    dif = abs((M - YAW_MONTAJE_DEG + 180) % 360 - 180)
    if dif > 5.0:
        print(f'  Ahora está en {YAW_MONTAJE_DEG:.0f}. Diferencia: {dif:.0f}°')
        print(f'  — que es exactamente lo que la cámara está girada')
        print(f'  respecto a la dirección de avance.')
    else:
        print(f'  Coincide con lo que ya tienes ({YAW_MONTAJE_DEG:.0f}): el')
        print('  montaje NO es el problema.')
    print('─' * 62)


class VerificadorCamara:
    """
    OJO CON LO QUE MIDE ESTO — me equivoqué al presentarlo.

    Compara (yaw_MoCap − YAW_MONTAJE_DEG) contra la dirección de avance.
    Pero el lazo de yaw fuerza justamente que esa resta siga a la
    tangente, así que YAW_MONTAJE_DEG se cancela y esto sale ~0° AUNQUE
    la cámara mire a 90°. Verifica que el LAZO CIERRA, no hacia dónde
    mira la cámara.

    Para lo segundo:  python volar_ruta.py --calibrar
    """

    V_MINIMA = 0.06      # m/s: por debajo, la dirección de avance es ruido

    def __init__(self):
        self.difs = []
        self.ult = None

    def muestra(self, x, y, t):
        if self.ult is not None:
            xa, ya, ta = self.ult
            dt = t - ta
            if dt > 1e-3:
                dx, dy = x - xa, y - ya
                if np.hypot(dx, dy) / dt >= self.V_MINIMA:
                    nariz = _yaw_mundo_del_mocap()
                    if nariz is not None:
                        self.difs.append(np.degrees(
                            _wrap(nariz - np.arctan2(dy, dx))))
        self.ult = (x, y, t)

    def informe(self, objetivo_deg=0.0):
        """
        objetivo_deg: ángulo que DEBERÍA haber entre la nariz y el avance.
        0 en modo tangente, ±90 en perpendicular.
        """
        if len(self.difs) < 20:
            print('  Cámara       : sin datos suficientes para verificar.')
            return
        d = np.radians(np.array(self.difs))
        # Media CIRCULAR: la aritmética de +179 y -179 daría 0, que es falso
        med = float(np.degrees(np.arctan2(np.sin(d).mean(), np.cos(d).mean())))
        disp = float(np.degrees(np.std(_wrap(d - np.radians(med)))))
        desvio = float(np.degrees(_wrap(np.radians(med - objetivo_deg))))

        print()
        print('  ── ¿HACIA DÓNDE MIRÓ LA CÁMARA? ──')
        print(f'  Ángulo nariz ↔ dirección de avance:')
        print(f'    medido {med:+.1f}°   pedido {objetivo_deg:+.1f}°   '
              f'→ desvío {desvio:+.1f}°')
        print(f'    dispersión {disp:.1f}°   ({len(self.difs)} muestras)')

        print('    ⚠ ESTO NO DICE HACIA DÓNDE MIRÓ LA CÁMARA.')
        print('      Compara (yaw_MoCap − YAW_MONTAJE_DEG) contra el avance,')
        print('      y el lazo fuerza justo esa resta, así que el montaje se')
        print('      cancela y sale ~0° aunque la nariz esté a 90°.')
        print('      Para saber hacia dónde mira:  --calibrar')

        a = abs(desvio)
        if a <= 25.0:
            # ANTES AQUÍ DECÍA «CORRECTO. La cámara encuadró lo que se
            # pidió». Era falso y costó un vuelo de confusión: el 17-sep
            # imprimió «desvío −0.6° → CORRECTO» con la nariz apuntando al
            # eje +X en una ruta que corría a lo largo de ∓Y. El docstring
            # de esta clase ya avisaba de que el montaje se cancela; el
            # mensaje impreso lo contradecía.
            print('  → El LAZO DE YAW CERRÓ: la nariz siguió a su referencia')
            print('    (desvío pequeño y constante). Si aun así la cámara')
            print('    apunta donde no debe, el que está mal es')
            print(f'    YAW_MONTAJE_DEG ({YAW_MONTAJE_DEG:.1f}), no el lazo.')
        elif a >= 155.0:
            print('  → JUSTO AL REVÉS (180°).')
            if YAW_MODO == 'perpendicular':
                otro = 'derecha' if objetivo_deg > 0 else 'izquierda'
                print( '    Está mirando hacia ADENTRO en vez de hacia afuera.')
                print(f"    ARREGLO: YAW_LADO = '{otro}'")
            else:
                print(f'    ARREGLO: YAW_MONTAJE_DEG {YAW_MONTAJE_DEG:.0f} → '
                      f'{(YAW_MONTAJE_DEG + 180.0) % 360.0:.0f}')
        else:
            print(f'  → DESVIADA {desvio:+.0f}° de lo pedido.')
            print(f'    Si el desvío es constante, es el montaje:')
            print(f'    YAW_MONTAJE_DEG {YAW_MONTAJE_DEG:.0f} → '
                  f'{(YAW_MONTAJE_DEG + desvio) % 360.0:.0f}')
            print(f'    Si quieres ese encuadre, es más simple ajustar')
            print(f'    YAW_ANGULO_DEG.')
        if disp > 30.0:
            print(f'    (dispersión {disp:.0f}°: en las vueltas la nariz se')
            print('     queda atrás. Normal si el giro es cerrado.)')


def ejecutar_ruta_carrot(scf, ruta_suave_2d, cultivo):
    """
    Vuela la ruta empujando una consigna de POSICIÓN que avanza sobre ella.
    Devuelve True si voló; False para que el vuelo caiga al método de
    siempre sin dejar al dron a medias.
    """
    try:
        import pursuit_ruta as pp_mod
    except Exception as e:
        print(f'⚠ No se pudo importar pursuit_ruta.py: {e}')
        return False

    altura = cultivo['altura']

    ruta = np.asarray(ruta_suave_2d, dtype=float)
    path = np.column_stack([ruta[:, 0], ruta[:, 1],
                            np.full(len(ruta), float(altura))])
    path = pp_mod.remuestrear(path, PASO_RUTA)

    pp = pp_mod.PurePursuit(tol_meta=TOL_META).set_path(path)
    v_prof = pp_mod.perfil_velocidad(pp.s, pp.kappa, pp.T,
                                     v_max=V_MAX, v_min=V_MIN,
                                     a_lat=A_LAT_MAX, a_tan=A_TAN_MAX)
    t_est = float(np.sum(np.diff(pp.s) / np.maximum(v_prof[:-1], 1e-3)))
    k_max = float(pp.kappa.max())

    print()
    print('─' * 62)
    print('  ZANAHORIA MÓVIL  (consigna de posición a %.0f Hz)' % CTRL_HZ)
    print('─' * 62)
    print(f'  Puntos de ruta   : {len(pp.path)}  ({pp.length:.2f} m)')
    print(f'  Radio mínimo     : {1.0/max(k_max,1e-6):.2f} m')
    print(f'  Velocidad        : {v_prof.min():.2f} – {v_prof.max():.2f} m/s')
    print(f'  Tiempo estimado  : {t_est:.1f} s')
    print(f'  Avance máximo    : {AVANCE_MAX*100:.0f} cm por delante del dron')
    print(f'  Yaw              : {"sigue la ruta" if SEGUIR_YAW_RUTA else "fijo"}')
    print('─' * 62)

    print('Esperando posicion inicial del MoCap...')
    t0 = time.time()
    while (P.mocap_pose['x'] == 0.0 and P.mocap_pose['y'] == 0.0
           and P.mocap_pose['z'] == 0.0):
        if P.stop_event.is_set():
            return True
        if time.time() - t0 > 10.0:
            print('Timeout esperando MoCap.')
            return False
        time.sleep(0.05)

    cmd = scf.cf.commander
    periodo = 1.0 / CTRL_HZ

    def pos_cmd(x, y, z, yaw_deg=YAW_FIJO_DEG):
        """
        MISMA GUARDA QUE EN LA VERSIÓN 1, y por el mismo motivo: el paro de
        emergencia llama a send_stop_setpoint() desde otro hilo. Si este
        lazo manda una consigna justo después, el dron REARMA los motores
        —una consigna de posición no es "apagado", es "vete ahí"— y el paro
        no para nada.
        """
        if P.stop_event.is_set():
            return False
        cmd.send_position_setpoint(float(x), float(y), float(z),
                                   float(yaw_deg))
        return True

    def vel_cmd(vx, vy, vz):
        if P.stop_event.is_set():
            return False
        cmd.send_velocity_world_setpoint(float(vx), float(vy), float(vz), 0.0)
        return True

    def ir_al_inicio(dst, max_v=0.20, kp=0.6, tol=0.04, timeout=15.0):
        """
        Copiada de ejecutar_ruta_pursuit (sistema_vuelo.py), que llega
        bien desde cualquier punto: velocidad corregida con el MoCap desde el
        PC. kp=0.6 en horizontal con tope 0.20 m/s y kp_z=0.8 con tope
        ±0.15 m/s; más ganancia no llega antes, llega oscilando.
        """
        print(f'Moviéndose al inicio: ({dst[0]:.2f}, {dst[1]:.2f}, {dst[2]:.2f})...')
        t0, kp_z = time.time(), 0.8
        t_anterior, hueco_max = None, 0.0
        while not P.stop_event.is_set():
            ex = dst[0] - P.mocap_pose['x']
            ey = dst[1] - P.mocap_pose['y']
            ez = dst[2] - P.mocap_pose['z']
            if np.hypot(ex, ey) < tol and abs(ez) < 0.06:
                break
            if time.time() - t0 > timeout:
                print('  ⚠ Timeout yendo al inicio — continuando.')
                break
            vx, vy = kp * ex, kp * ey
            h = np.hypot(vx, vy)
            if h > max_v:
                vx *= max_v / h
                vy *= max_v / h
            t_envio = time.perf_counter()
            if vel_cmd(vx, vy, float(np.clip(kp_z * ez, -0.15, 0.15))):
                if t_anterior is not None:
                    hueco_max = max(hueco_max, t_envio - t_anterior)
                t_anterior = t_envio
            time.sleep(0.02)
        vel_cmd(0.0, 0.0, 0.0)
        time.sleep(0.1)
        print(f'  Posición alcanzada: ({P.mocap_pose["x"]:.3f}, '
              f'{P.mocap_pose["y"]:.3f}, {P.mocap_pose["z"]:.3f})')
        aviso = (' — ADVERTENCIA: el bucle se atrasó y el dron puede oscilar'
                 if hueco_max > 0.100 else '')
        print(f'  aproximación al inicio: hueco máximo entre consignas '
              f'{hueco_max * 1000:.1f} ms{aviso}')
        return not P.stop_event.is_set()

    def mantener(objetivo, duracion, kp=1.5, kp_z=2.0, v_max=0.35):
        """
        Copiada de ejecutar_ruta_pursuit: hover ANCLADO al punto con
        velocidad corregida con el MoCap. Mandar velocidad cero no mantiene
        la posición: deja al dron a la deriva.
        """
        t0 = time.time()
        while time.time() - t0 < duracion and not P.stop_event.is_set():
            vx = kp * (objetivo[0] - P.mocap_pose['x'])
            vy = kp * (objetivo[1] - P.mocap_pose['y'])
            vz = kp_z * (objetivo[2] - P.mocap_pose['z'])
            h = np.hypot(vx, vy)
            if h > v_max:
                vx *= v_max / h
                vy *= v_max / h
            vel_cmd(vx, vy, float(np.clip(vz, -0.3, 0.3)))
            time.sleep(periodo)
        return not P.stop_event.is_set()

    def quieto(segundos, x, y, z, yaw_deg, motivo=''):
        """
        Mantiene la consigna sin moverse. Dos cosas a la vez: deja que los
        transitorios de la fase anterior se apaguen antes de empezar la
        siguiente, y —tan importante como eso— SIGUE MANDANDO consignas.
        Un hueco sin consignas hace saltar el watchdog del commander, y de
        ahí se sale con un tirón.
        """
        if segundos <= 0:
            return not P.stop_event.is_set()
        if motivo:
            print(f'  · {segundos:.1f} s quieto ({motivo})')
        t0 = time.time()
        while time.time() - t0 < segundos and not P.stop_event.is_set():
            pos_cmd(x, y, z, yaw_deg)
            time.sleep(periodo)
        return not P.stop_event.is_set()

    # La carga del modelo no debe competir con las consignas en vuelo.
    if EVASION_MODO in ('observar', 'esquivar'):
        vis = P.VISION_EN_CURSO
        if vis is None:
            print('  No se despega: evasión activa sin sistema de visión.')
            return True
        if not _esperar_vision_lista(vis, VISION_ESPERA_S):
            return True          # no caer al go_to, que despegaría igual

    # ── Motive tiene que VER el dron donde está apoyado ──
    # Si no, el puente repite la última pose vista (en la mano) y todo lo que
    # sigue —reset del EKF, yaw, punto de despegue— parte de una posición
    # falsa. Era la causa de "solo despega recto en un punto exacto".
    print('Comprobando que el MoCap rastrea el dron donde está...')
    if not _esperar_mocap_vivo(MOCAP_VIVO_ESPERA_S):
        return True          # True: que no caiga al go_to, que despegaría igual

    # ── El yaw del EKF debe coincidir con el real ANTES de dejar el suelo ──
    # Se reinicia el EKF con el rumbo real del dron (da igual cómo se puso) y
    # se comprueba; si no queda alineado, no se despega.
    # Desactivado (ALINEAR_YAW_ANTES_DE_DESPEGAR): el pursuit, que despega
    # bien, no reinicia el EKF antes de subir.
    if ALINEAR_YAW_ANTES_DE_DESPEGAR:
        print('Alineando el yaw del EKF con el rumbo real del dron...')
        _alinear_yaw_ekf(scf)
        print('Comprobando la alineación del yaw del EKF...')
        if not _esperar_yaw_alineado(scf, ALINEA_YAW_TOL_DEG, ALINEA_YAW_ESPERA_S):
            return True      # True: que no caiga al go_to, que despegaría igual

    if not _esperar_ekf_listo(scf, EKF_ESPERA_MIN_S, EKF_ESPERA_MAX_S,
                              EKF_TOL_M, EKF_ESTABLE_S):
        return True          # no caer al go_to, que despegaría igual

    # ── Despegue: VERTICAL PURO, como ejecutar_ruta_pursuit ──
    # vx = vy = 0 hasta la altura: mientras haya contacto con el suelo, una
    # corrección horizontal es deslizar el dron contra el suelo.
    # Red de seguridad: si se aleja de lado más de DESPEGUE_DESVIO_MAX, paro
    # de emergencia (antes salía del Robotat). También si el MoCap se queda
    # sin muestras nuevas más de MOCAP_TIMEOUT: pasó una vez (25-sep-2026) que
    # se quedó CONGELADO 3.6 s completos —sin refrescarse, no un salto— y el
    # bucle, ciego a eso, siguió empujando velocidad vertical sin corregir de
    # lado; cuando el MoCap volvió, el dron ya había derivado de verdad los
    # 65 cm que disparó la otra red. Aquí se para ANTES de que eso se acumule.
    print(f'Despegando (low-level) hasta z={altura:.2f} m...')
    x_ini, y_ini = P.mocap_pose['x'], P.mocap_pose['y']
    reg_despegue = []
    t0 = time.time()
    while not P.stop_event.is_set():
        x, y, z = P.mocap_pose['x'], P.mocap_pose['y'], P.mocap_pose['z']
        pose = getattr(scf, 'pose_data', None) or {}
        reg_despegue.append((round(time.time() - t0, 3), x, y, z,
                             pose.get('roll'), pose.get('pitch'), pose.get('yaw')))
        if z >= altura - 0.05 or time.time() - t0 > 10.0:
            break
        # Sin CAMBIOS en la pose (no solo sin mensajes): un MoCap congelado
        # sigue mandando mensajes y el vigía de llegada no lo ve.
        edad_mocap = time.perf_counter() - P._t_mocap_cambio[0]
        if edad_mocap > MOCAP_TIMEOUT:
            P.activar_paro_emergencia(
                f'despegue: {edad_mocap:.2f} s sin que el MoCap vea moverse al dron')
            break
        desvio = float(np.hypot(x - x_ini, y - y_ini))
        if desvio > DESPEGUE_DESVIO_MAX:
            P.activar_paro_emergencia(
                f'despegue: el dron se alejó {desvio * 100:.0f} cm de lado')
            break
        vel_cmd(0.0, 0.0, DESPEGUE_VZ)
        time.sleep(0.02)
    _guardar_registro_despegue(reg_despegue)
    vel_cmd(0.0, 0.0, 0.0)
    time.sleep(0.1)
    print(f'  Altura alcanzada: z={P.mocap_pose["z"]:.3f} m')
    if P.stop_event.is_set():
        return True

    # ── Yaw de crucero de la aproximación ──
    # ANTES: pos_cmd() usaba su valor por defecto, YAW_FIJO_DEG = 0°. O sea
    # que la PRIMERA consigna de posición del vuelo le pedía al dron ponerse
    # a 0° en el marco del EKF, de golpe y sin límite de velocidad angular.
    # Ese era el primero de los tres giros: un giro que no hacía falta, para
    # llegar a una orientación que tampoco hacía falta, porque cuatro
    # segundos después había que girar otra vez hacia el primer tramo.
    #
    # Ahora se lee el yaw que el dron YA tiene y se le manda ese mismo: la
    # consigna coincide con la realidad, así que no hay giro. La nariz se
    # queda quieta durante toda la aproximación y solo gira UNA vez, al
    # final, hacia el primer tramo.
    _ye = _yaw_ekf(scf)
    yaw_crucero = float(np.degrees(_ye)) if _ye is not None else YAW_FIJO_DEG
    if _ye is None:
        print('  (sin yaw del dron todavía: se aproxima con yaw fijo)')

    # ── Al inicio de la ruta: igual que ejecutar_ruta_pursuit ──
    # Velocidad corregida con el MoCap desde el PC, y 1 s anclado al punto.
    # Desde el giro de la nariz en adelante, consignas de posición.
    t_preambulo = time.time()
    if not ir_al_inicio((path[0][0], path[0][1], altura)):
        return True

    print('Estabilizando sobre el punto inicial (1.0 s)...')
    if not mantener((path[0][0], path[0][1], altura), 1.0):
        return True

    pp.reset()
    pp.step(P.mocap_pose['x'], P.mocap_pose['y'])

    # ── De qué lado mira la cámara ──
    lado, area = _lado_hacia_afuera(pp.path)
    if YAW_LADO == 'izquierda':
        lado = +1
    elif YAW_LADO == 'derecha':
        lado = -1
    if YAW_MODO == 'perpendicular':
        sentido = 'antihorario' if area > 0 else 'horario'
        print(f'  Cámara a {YAW_ANGULO_DEG:.0f}° del avance, hacia la '
              f'{"IZQUIERDA" if lado > 0 else "DERECHA"}')
        if YAW_LADO == 'auto':
            print(f'    (la ruta se recorre en sentido {sentido}, así que ese')
            print( '     lado es el que mira HACIA AFUERA del área)')

    # ── Calibrar el yaw ANTES de arrancar, con el dron ya quieto ──
    # La calibración tarda ~1.5 s. Antes no mandaba NINGUNA consigna en ese
    # rato: el commander del firmware se quedaba sin nada que seguir, y al
    # volver a recibir consignas el dron pegaba un tirón. Ahora se reenvía
    # la consigna de hold en cada vuelta, igual que hace la medición del
    # fondo de disparidad.
    yawc = ControlYaw()
    girar = SEGUIR_YAW_RUTA and yawc.calibrar(
        scf, mantener=lambda: pos_cmd(path[0][0], path[0][1], altura,
                                      yaw_crucero))
    if girar:
        # ── EL ÚNICO GIRO DEL PREÁMBULO ──
        # Alinear la nariz antes de empezar a avanzar: entrar a la ruta
        # girando arrastra la posición.
        #
        # Antes esto giraba durante un TIEMPO FIJO calculado como
        # giro/YAW_RATE_MAX, o sea justo el tiempo teórico y ni un ciclo
        # más: 90° a 45 °/s = 2.0 s exactos. Con el límite de velocidad
        # angular de la referencia, ese tiempo NUNCA alcanza, así que el
        # giro se quedaba a medias y lo terminaba el lazo de la ruta ya en
        # marcha. Eso se ve como un segundo giro, y encima mientras el dron
        # ya avanza. Ahora se gira hasta CONVERGER, con tope de tiempo, y
        # más despacio (YAW_RATE_APROX), que es un giro en el sitio y no
        # hay ninguna prisa.
        _, T0, _ = pp.estado_en(min(pp.s_proj + YAW_LOOKAHEAD, pp.length))
        objetivo = _yaw_objetivo(T0, lado)
        ym = _yaw_mundo_del_mocap()
        giro = abs(np.degrees(_wrap(objetivo - ym))) if ym is not None else 0.0
        # La referencia arranca donde está la nariz AHORA, no en el
        # objetivo: así el primer ciclo no manda un escalón.
        if ym is not None:
            yawc.fijar(ym)
        t_prev = max(giro / YAW_RATE_APROX, 0.5) + 1.0
        print(f'  Alineando la nariz con el primer tramo: {giro:.0f}° '
              f'a {YAW_RATE_APROX:.0f}°/s (~{t_prev:.1f} s)...')
        t0 = time.time()
        e_deg = giro
        yaw_cmd_giro = yaw_crucero    # último yaw mandado, para la pausa
        while not P.stop_event.is_set():
            yawc.actualizar_desfase(scf)
            yd, err = yawc.paso(objetivo, periodo,
                                rate_max=YAW_RATE_APROX,
                                acel_max=YAW_ACEL_APROX)
            yaw_cmd_giro = yd
            pos_cmd(path[0][0], path[0][1], altura, yd)
            ym = _yaw_mundo_del_mocap()
            if ym is not None:
                e_deg = abs(np.degrees(_wrap(objetivo - ym)))
            t_giro = time.time() - t0
            # Se para cuando la nariz LLEGÓ, no cuando se acabó el reloj.
            if t_giro > 0.5 and e_deg < YAW_TOL_APROX:
                print(f'    alineada en {t_giro:.1f} s (error {e_deg:.1f}°)')
                break
            if t_giro > YAW_T_MAX_APROX:
                print(f'    ⚠ {YAW_T_MAX_APROX:.0f} s y la nariz sigue a '
                      f'{e_deg:.0f}°. Se arranca igual: el lazo de la ruta '
                      'la termina de llevar.')
                break
            time.sleep(periodo)
        if P.stop_event.is_set():
            return True

    # Quieto también DESPUÉS de girar: si no, el primer paso de la ruta se
    # encadena con el final del giro y vuelve a haber dos cosas a la vez.
    if not quieto(APROX_PAUSA, path[0][0], path[0][1], altura,
                  yaw_cmd_giro if girar else yaw_crucero,
                  'antes de arrancar la ruta'):
        return True

    # El preámbulo suave cuesta tiempo de vuelo, y el tiempo de vuelo es
    # batería. Se imprime para que sea un dato y no una sorpresa: si sale
    # muy largo, el que manda casi siempre es APROX_V_MAX.
    print(f'  Preámbulo (aproximación + alineación): '
          f'{time.time() - t_preambulo:.1f} s')

    P.led_on(scf)

    # ══════════════════════════════════════════════════════════
    # LAZO DE LA ZANAHORIA
    # ══════════════════════════════════════════════════════════
    print(f'Siguiendo la ruta ({t_est:.0f} s estimados)...')
    P.VENTANA_RUTA['ini'] = time.time()

    s_car = float(pp.s_proj)          # arco de la zanahoria
    # ── EL TERCER GIRO ESTABA AQUÍ ──
    # Esto era `yaw_cmd_deg = YAW_FIJO_DEG`, o sea 0°. Y dos líneas más
    # abajo, la medición del fondo de disparidad reenvía la consigna con
    # ESE yaw durante dos segundos. Resultado: el dron acababa de girar 90°
    # para alinearse con el primer tramo, e inmediatamente se le mandaba
    # volver a 0°, para que el lazo de la ruta lo devolviera a 90° dos
    # segundos después. Tres giros donde hacía falta uno, y los dos últimos
    # a la velocidad que quisiera el firmware, sin límite ninguno.
    #
    # Ahora arranca donde de verdad está la nariz: si el lazo de yaw está
    # activo, en su referencia; si no, en el yaw de crucero, que es el que
    # el dron ya tiene. Ninguno de los dos manda moverse.
    if girar and yawc.ref is not None and yawc.desfase is not None:
        yaw_cmd_deg = float(np.degrees(_wrap(yawc.ref - yawc.desfase)))
    else:
        yaw_cmd_deg = yaw_crucero
    err_yaw = 0.0
    verif = VerificadorCamara()
    evas = Evasion(_VISION['obj'])
    evas.ruta_xy = path[:, :2]      # para el filtro por zona
    evas.preparar()
    maniobra = ManiobraEvasion(
        holgura=EVASION_HOLGURA, desvio_max=EVASION_DESVIO_MAX,
        radio_influencia=EVASION_RADIO_INF, radio_seguro=EVASION_RADIO_SEG,
        cola=EVASION_COLA, compromiso_min=EVASION_COMPROMISO_MIN,
        v_max=min(V_MAX, APROX_V_MAX),
        a_max=min(A_LAT_MAX, A_TAN_MAX, APROX_A_MAX), t_min=APROX_T_MIN,
        tol_pos=APROX_TOL, tol_yaw=np.radians(YAW_TOL_APROX),
        edad_max=EVASION_VISTA_MAX_S, espera_max=YAW_T_MAX_APROX,
        espera_vista_max=MANIOBRA_ESPERA_VISTA_S,
        margen_lado=EVASION_MARGEN_LADO)
    estado_maniobra_impreso = None

    # ── COMPUERTA: se mide el fondo del laboratorio AHORA ────────────
    # El dron ya está quieto sobre el punto inicial y con la nariz puesta,
    # que es exactamente la situación que hace falta: sin nada delante.
    # Son dos segundos que hasta ahora no se aprovechaban para nada.
    if (evas.activa and EVASION_DISP_AUTO and evas.compuerta is None):
        print('  [evasión] midiendo el fondo de disparidad '
              f'({EVASION_DISP_AUTO_SEG:.0f} s, no pongas nada delante)...')
        evas.calibrar_fondo(
            mantener=lambda: pos_cmd(path[0][0], path[0][1], altura,
                                     yaw_cmd_deg))
        if evas.compuerta is None:
            print('  ⚠ [evasión] sin compuerta. Con EVASION_MODO=\'esquivar\'')
            print('    esto es un generador de ruido: cualquier columna por')
            print('    encima de la mediana cuenta como obstáculo. Vuela en')
            print('    \'observar\' hasta tener el número.')
    registro, n_ciclos, n_stale, n_parado = [], 0, 0, 0
    motivo_fin = 'ruta completada'
    t_ini = time.perf_counter()
    t_rampa = t_ini
    t_next, t_print = t_ini, t_ini
    timeout = FACTOR_TIMEOUT * t_est + 8.0
    ultimo = None
    tiempo_maniobra = 0.0
    t_ciclo_anterior = t_ini
    registro_maniobra = []

    def registrar_maniobra(t):
        diag = getattr(maniobra, 'diag', {})
        yaw_mocap = _yaw_mundo_del_mocap()      # mismo marco que yaw_obj
        estable = diag.get('yaw_estable')
        fila = [t-t_ini, maniobra.estado,
                float(P.mocap_pose['x']), float(P.mocap_pose['y']),
                float(P.mocap_pose['z']),
                float(np.degrees(yaw_mocap)) if yaw_mocap is not None else None,
                float(np.degrees(maniobra.yaw_obj)),
                float(maniobra.fijo[0]), float(maniobra.fijo[1]),
                diag.get('err_yaw_deg'), diag.get('err_pos_cm'),
                bool(estable) if estable is not None else None,
                diag.get('edad_vista_s')]
        # Completar la misma fila tras paso; conservarla si el ciclo sale antes.
        if registro_maniobra and registro_maniobra[-1][0] == t-t_ini:
            registro_maniobra[-1] = fila
        else:
            registro_maniobra.append(fila)

    try:
        while not P.stop_event.is_set():
            t_now = time.perf_counter()
            if maniobra.activa:
                registrar_maniobra(t_now)
            if maniobra.activa and maniobra.estado != 'BLOQUEADA':
                tiempo_maniobra += t_now - t_ciclo_anterior
            t_ciclo_anterior = t_now
            critica_desde = P._bateria_estado['critica_desde']
            if (critica_desde is not None
                    and time.time() - critica_desde > BATERIA_ATERRIZA_S):
                print('⚠ Batería crítica sostenida — terminando la ruta para aterrizar.')
                motivo_fin = 'bateria critica'
                break
            if t_now - t_ini - tiempo_maniobra > timeout:
                print('⚠ Timeout de vuelo — abortando el seguimiento.')
                break

            # Perro guardián: sin MoCap fresco no se avanza la zanahoria.
            # Se repite la última consigna → el dron se queda flotando.
            edad_mocap = t_now - P._t_mocap_fresco[0]
            if edad_mocap > MOCAP_TIMEOUT:
                n_stale += 1
                # ══ FLOTAR NO PUEDE SER LA RESPUESTA PARA SIEMPRE ══
                # Quedarse repitiendo la última consigna está bien durante
                # un hueco corto del MoCap: el dron se para solo y espera.
                # Pero si el MoCap NO VUELVE, este bucle seguía repitiendo
                # la consigna hasta el timeout de vuelo —minutos— con el
                # dron en el aire y sin saber dónde está. Nadie lo bajaba.
                #
                # Pasado MOCAP_ATERRIZA_S se baja a ciegas. Es peor que
                # aterrizar con MoCap y mucho mejor que agotar la batería
                # flotando sin referencia.
                if edad_mocap > MOCAP_ATERRIZA_S:
                    print(f'\n  ⚠ {edad_mocap:.1f} s sin MoCap. Bajando a '
                          f'ciegas: no se sabe dónde está el dron.')
                    motivo_fin = 'sin MoCap'
                    break
                if ultimo is not None:
                    pos_cmd(*ultimo)
                time.sleep(periodo)
                continue

            x, y, z = P.mocap_pose['x'], P.mocap_pose['y'], P.mocap_pose['z']

            # La evasión decide ANTES del avance de ruta y del yaw nominal.
            # Durante la maniobra no se desplaza la zanahoria ni se libera lado.
            p_nom, T_nom, _ = pp.estado_en(s_car)
            if evas.activa:
                evas.paso((x, y), _yaw_mundo_del_mocap(), p_nom[:2], T_nom, t_now)
            if evas.esquiva and evas.activa and evas.mapa is not None:
                obst, pesos = evas.mapa.confirmados(EVASION_PESO_MIN)
                if not maniobra.activa:
                    inicio = ultimo[:2] if ultimo is not None else (x, y)
                    maniobra.iniciar(inicio, p_nom[:2], T_nom, obst, pesos,
                                     t_now, lado=evas.lado_comp)
                    if maniobra.activa:
                        evas.mantener_compromiso = True
                        evas.lado_comp = maniobra.lado
            if maniobra.activa:
                if not girar:
                    maniobra.bloquear('La maniobra necesita control de yaw activo', t_now)
                obst = (evas.mapa.confirmados(EVASION_PESO_MIN)[0]
                        if evas.mapa is not None else np.empty((0, 2)))
                # Retorno local hacia delante: evita saltar a un surco vecino.
                ventana_retorno = 2*(EVASION_RADIO_INF + EVASION_HOLGURA)
                indices = np.flatnonzero((pp.s >= pp.s_proj)
                                        & (pp.s <= pp.s_proj + ventana_retorno))
                j = int(indices[np.argmin(np.linalg.norm(
                    pp.path[indices, :2] - (x, y), axis=1))]) if len(indices) else pp.index
                salida = maniobra.paso(
                    (x, y), _yaw_mundo_del_mocap(), t_now,
                    evas.vista if evas.activa else None, obst, pp.path[j, :2],
                    yaw_estable=girar and abs(yawc.vel) < np.radians(0.5))
                registrar_maniobra(t_now)
                if (maniobra.estado == 'BLOQUEADA'
                        and t_now - maniobra.desde > MANIOBRA_BLOQUEO_ATERRIZA_S):
                    print(f'⚠ Maniobra bloqueada — aterrizando: {maniobra.motivo}')
                    motivo_fin = 'maniobra bloqueada: ' + maniobra.motivo
                    break
                if (maniobra.estado, maniobra.motivo) != estado_maniobra_impreso:
                    print(f'  [maniobra] {maniobra.estado} {maniobra.motivo}')
                    estado_maniobra_impreso = (maniobra.estado, maniobra.motivo)
                if salida is not None:
                    objetivo_xy, objetivo_yaw = salida
                    if girar:
                        yawc.actualizar_desfase(scf)
                        yaw_cmd_deg, err_yaw = yawc.paso(
                            objetivo_yaw, periodo,
                            rate_max=min(YAW_RATE_MAX, YAW_RATE_APROX),
                            acel_max=min(YAW_ACEL_MAX, YAW_ACEL_APROX))
                    ultimo = (*objetivo_xy, altura, yaw_cmd_deg)
                    if not pos_cmd(*ultimo):
                        break
                    time.sleep(periodo)
                    t_next = time.perf_counter()
                    continue
                evas.mantener_compromiso = False
                evas._soltar_lado()
                evas.desvio_suave = 0.0
                pp.step(x, y, max_avance=ventana_retorno)
                s_car = pp.s_proj
                t_rampa = t_now

            pp.step(x, y, max_avance=0.06 + 2.5 * V_MAX * periodo)
            if pp.finished:
                break

            # Velocidad del perfil en el punto donde está la zanahoria
            v_obj = float(np.interp(s_car, pp.s, v_prof))

            # ── GIRO: se lee la tangente un poco POR DELANTE de la
            #    zanahoria y se le aplica el ángulo pedido (0° = hacia
            #    adelante, ±90° = perpendicular, mirando hacia afuera).
            factor_yaw = 1.0
            if girar:
                yawc.actualizar_desfase(scf)
                _, T_mira, _ = pp.estado_en(min(s_car + YAW_LOOKAHEAD,
                                                pp.length))
                yaw_cmd_deg, err_yaw = yawc.paso(
                    _yaw_objetivo(T_mira, lado), periodo)

                # ACOPLE NARIZ→VELOCIDAD. Mismo principio que el anclaje de
                # la zanahoria, aplicado al giro: si la nariz va retrasada
                # respecto a la referencia, la zanahoria afloja y la espera.
                # Sin esto, en las vueltas en U el dron llega al siguiente
                # surco todavía girando.
                # El suelo YAW_FACTOR_MIN es lo que garantiza que NO se
                # pare: simulado, la velocidad mínima en régimen queda en
                # 0.16-0.18 m/s.
                exceso = max(abs(np.degrees(err_yaw)) - YAW_BANDA_MUERTA, 0.0)
                factor_yaw = float(np.clip(1.0 - exceso / YAW_ERR_AFLOJA,
                                           YAW_FACTOR_MIN, 1.0))

            # ── EL ANCLAJE: la zanahoria avanza con el tiempo, PERO no
            #    puede separarse más de AVANCE_MAX de la proyección real
            #    del dron. Si el dron se retrasa, le espera. Esto es lo
            #    que start_trajectory no tenía.
            # Zanahoria adaptada a la curvatura (1.0 = desactivado).
            # Se interpola entre AVANCE_MAX en recta y AVANCE_MAX*K en la
            # curva más cerrada de la ruta.
            if abs(AVANCE_CURVA_K - 1.0) > 1e-6:
                mezcla = min(pp.curvatura_aqui() / AVANCE_CURVA_REF, 1.0)
                avance = AVANCE_MAX * (1.0 + (AVANCE_CURVA_K - 1.0) * mezcla)
                avance = float(np.clip(avance, AVANCE_MIN + 0.01, 0.40))
            else:
                avance = AVANCE_MAX

            r = _rampa_inicio(t_now, t_rampa, RUTA_RAMPA_S)
            s_car = min(s_car + v_obj * factor_yaw * periodo * r, pp.s_proj + avance * r)
            s_car = max(s_car, pp.s_proj + AVANCE_MIN * r)
            s_car = min(s_car, pp.length)

            p_car, T_car, kappa = pp.estado_en(s_car)

            # Fuera de la máquina se sigue la ruta nominal, sin sumar
            # por segunda vez el desvío geométrico de la evasión.
            px_car, py_car = float(p_car[0]), float(p_car[1])
            ultimo = (px_car, py_car, altura, yaw_cmd_deg)
            if not pos_cmd(*ultimo):
                break

            verif.muestra(x, y, t_now)
            n_ciclos += 1
            registro.append((t_now - t_ini, pp.e_lat_signed,
                             s_car - pp.s_proj, v_obj, kappa,
                             np.degrees(err_yaw), factor_yaw))

            if t_now - t_print > 1.5:
                extra = (f'  nariz={np.degrees(err_yaw):+5.1f}°' if girar else '')
                print(f'  [{pp.progreso():5.1f}%] v={v_obj*factor_yaw:.2f} '
                      f'zanahoria={-(pp.s_proj - s_car)*100:5.1f} cm | '
                      f'e_lat={pp.e_lat_signed*100:+6.1f} cm  '
                      f'e_z={(altura - z)*100:+6.1f} cm{extra}')
                t_print = t_now

            t_next += periodo
            dorm = t_next - time.perf_counter()
            if dorm > 0:
                time.sleep(dorm)
            else:
                t_next = time.perf_counter()

        P.VENTANA_RUTA['fin'] = time.time()

        t_vuelo = time.perf_counter() - t_ini
        hz = n_ciclos / t_vuelo if t_vuelo > 0 else 0.0
        print(f'\n✓ Seguimiento terminado en {t_vuelo:.1f} s '
              f'({hz:.0f} Hz reales, {pp.progreso():.0f} % de la ruta)')
        if motivo_fin != 'ruta completada':
            print(f'  ⚠ El seguimiento terminó por: {motivo_fin}')
        if registro_maniobra:
            _guardar_registro_maniobra(registro_maniobra)
        if n_stale:
            print(f'  ⚠ {n_stale} ciclos con el MoCap viejo: se repitió la')
            print( '    última consigna y el dron se quedó flotando.')
        if n_parado:
            print(f'  · {n_parado} ciclos parados por la evasión '
                  f'({n_parado*periodo:.1f} s). Durante ellos la zanahoria y')
            print( '    la referencia de nariz quedaron CONGELADAS, así que al')
            print( '    soltar la parada no hubo escalón en la consigna.')

        if registro:
            r = np.array(registro)
            el = np.abs(r[:, 1])
            av = r[:, 2]
            print()
            print(f'  Error lateral : medio {el.mean()*100:5.2f} cm | '
                  f'máx {el.max()*100:5.2f} cm | '
                  f'p95 {np.percentile(el,95)*100:5.2f} cm')
            print(f'  Zanahoria     : media {av.mean()*100:5.1f} cm por delante '
                  f'(tope {AVANCE_MAX*100:.0f})')
            if av.mean() > 0.92 * AVANCE_MAX:
                print('    Casi siempre en el tope: el dron podría ir más')
                print('    rápido. Sube AVANCE_MAX.')
            elif av.mean() < 0.5 * AVANCE_MAX:
                print('    Muy por debajo del tope: el dron no alcanza la')
                print('    consigna. Baja V_MAX, no subas AVANCE_MAX.')

            if girar and r.shape[1] >= 7:
                ey = np.abs(r[:, 5])
                fy = r[:, 6]
                print(f'  Nariz         : error medio {ey.mean():5.1f}° | '
                      f'máx {ey.max():5.1f}° | p95 {np.percentile(ey,95):5.1f}°')
                frenadas = float((fy < 0.99).mean() * 100.0)
                print(f'  Acople giro   : afloja el {frenadas:.0f} % del tiempo, '
                      f'factor mínimo {fy.min():.2f}')
                if ey.mean() > 25.0:
                    print('    La nariz no sigue a la referencia. Baja')
                    print('    YAW_RATE_MAX o V_MAX: está girando más de lo')
                    print('    que puede.')
                if fy.min() <= YAW_FACTOR_MIN + 1e-6:
                    print(f'    Llegó al suelo del acople ({YAW_FACTOR_MIN}).')
                    print('    Ahí es donde más despacio fue por el giro; si')
                    print('    se notó un frenazo, sube YAW_FACTOR_MIN.')
                if frenadas > 40.0:
                    print(f'    Afloja demasiado. Sube YAW_BANDA_MUERTA '
                          f'({YAW_BANDA_MUERTA:.0f}° → '
                          f'{YAW_BANDA_MUERTA+10:.0f}°): te está frenando')
                    print('    por errores de nariz que no molestan.')

                obj = (lado * YAW_ANGULO_DEG
                       if YAW_MODO == 'perpendicular' else 0.0)
                verif.informe(objetivo_deg=obj)
            evas.t_vuelo = t_vuelo
            evas.informe()
    finally:
        P.led_off(scf)

    if P.stop_event.is_set():
        return True

    # ── Descenso en el sitio: última consigna XY, o posición actual del MoCap ──
    print('Aterrizando suavemente...')
    x_at = ultimo[0] if ultimo is not None else P.mocap_pose['x']
    y_at = ultimo[1] if ultimo is not None else P.mocap_pose['y']
    t0 = time.time()
    z_obj = altura
    while P.mocap_pose['z'] > 0.10 and not P.stop_event.is_set():
        if time.time() - t0 > 15.0:
            print('  ⚠ Timeout en el descenso.')
            break
        z_obj = max(z_obj - 0.15 * periodo, 0.0)
        if not pos_cmd(x_at, y_at, z_obj, yaw_cmd_deg):
            break
        time.sleep(periodo)
    if not P.stop_event.is_set():
        for _ in range(10):
            if not vel_cmd(0.0, 0.0, 0.0):
                break
            time.sleep(0.02)
        try:
            cmd.send_stop_setpoint()
        except Exception:
            pass
    print('✓ En el suelo.')
    return True


# Se inyecta en el módulo base: main() llama a ejecutar_ruta_pursuit cuando
# MODO_EJECUCION == 'pursuit', así que se sustituye por la de aquí. La
# original queda intacta en su archivo.
P.ejecutar_ruta_pursuit = ejecutar_ruta_carrot


def banner():
    print()
    print('═' * 62)
    print('  VERSIÓN 3 — ZANAHORIA MÓVIL (consigna de posición)')
    print('═' * 62)
    print('  Planificador     : el PC decide DÓNDE, el dron decide CÓMO')
    print('  Lazo de posición : contra el EKF del dron (a 500 Hz internos)')
    print('  Geometría        : calculada aquí contra el MoCap real')
    print(f'  Avance máximo    : {AVANCE_MAX*100:.0f} cm por delante del dron')
    print(f'  Consigna         : {CTRL_HZ:.0f} Hz')
    if SEGUIR_YAW_RUTA and YAW_MODO == 'perpendicular':
        print(f'  Cámara           : PERPENDICULAR al avance '
              f'({YAW_ANGULO_DEG:.0f}°), hacia AFUERA')
        print(f'                     lado: {YAW_LADO}  ·  máx {YAW_RATE_MAX:.0f}°/s')
    elif SEGUIR_YAW_RUTA:
        print(f'  Cámara           : HACIA ADELANTE, máx {YAW_RATE_MAX:.0f}°/s, '
              f'mirando {YAW_LOOKAHEAD*100:.0f} cm por delante')
    else:
        print('  Cámara           : nariz FIJA')
    if SEGUIR_YAW_RUTA:
        print( '                     El desfase EKF↔mundo se mide al arrancar')
        print( '                     y se sigue corrigiendo en vuelo.')
    print('─' * 62)
    print('  Si el lazo del PC se atasca, el dron se queda FLOTANDO')
    print('  (con consignas de velocidad se iría volando recto).')
    print('═' * 62)
    print()
    sat = V_MAX / 2.0     # KP_POS del controlador de posición ≈ 2.0
    print('  QUÉ TOCAR SEGÚN LO QUE VEAS:')
    print(f'    va lento .................... sube V_MAX (NO AVANCE_MAX:')
    print(f'                                  ya está en {AVANCE_MAX:.2f} y satura')
    print(f'                                  en {sat:.2f} m)')
    print('    corta las curvas por dentro . baja A_LAT_MAX 0.35 → 0.25')
    print('    oscila alrededor de la ruta . baja V_MAX')
    print('    la nariz llega tarde al surco YAW_RATE_MAX 45 → 70')
    print('    frena al girar .............. sube YAW_BANDA_MUERTA 15 → 25')
    print('    la cámara mira mal .......... el informe te dice el ángulo')
    print('                                  exacto y el YAW_MONTAJE_DEG')
    print('    quiero medir sin giro ....... SEGUIR_YAW_RUTA = False')
    print('    tarda mucho en ir al inicio . sube APROX_V_MAX '
          f'({APROX_V_MAX:.2f} m/s)')
    print('    el arranque sigue brusco .... baja APROX_A_MAX '
          f'({APROX_A_MAX:.2f} m/s²)')
    print('    el giro previo va muy lento . sube YAW_RATE_APROX '
          f'({YAW_RATE_APROX:.0f} °/s)')
    print('    el giro previo sigue brusco . baja YAW_ACEL_APROX '
          f'({YAW_ACEL_APROX:.0f} °/s²)')
    print('    quiero más pausa entre fases  sube APROX_PAUSA '
          f'({APROX_PAUSA:.1f} s)')
    print('    esquiva muy lejos / muy cerca EVASION_HOLGURA '
          f'({EVASION_HOLGURA*100:.0f} cm;')
    print(f'                                  con cinta salen ~'
          f'{EVASION_HOLGURA*65:.0f} cm)')
    print('    hace eses y no lo rodea ..... sube EVASION_COMPROMISO_MIN')
    print(f'                                  ({EVASION_COMPROMISO_MIN*100:.0f} cm)')
    print('    vuelve a la ruta muy pronto . sube EVASION_COLA '
          f'({EVASION_COLA*100:.0f} cm)')
    print()
    print('  EL EXPERIMENTO PENDIENTE (2 vuelos, zanja una duda real):')
    print('    AVANCE_CURVA_K = 0.7  → un vuelo, anota el RMS')
    print('    AVANCE_CURVA_K = 1.4  → otro vuelo, anota el RMS')
    print('    El que gane dice si en tus curvas de R=8 cm manda el')
    print('    recorte por cuerda o el retardo del lazo. No lo puedo')
    print('    simular: mi modelo del dron es demasiado optimista.')
    if AVANCE_MAX < sat:
        print()
        print(f'    ⚠ AVANCE_MAX ({AVANCE_MAX:.2f}) está por debajo de')
        print(f'      V_MAX/2 = {sat:.2f}: la zanahoria te está limitando')
        print(f'      la velocidad a ~{AVANCE_MAX*2.0:.2f} m/s.')
    print()


if __name__ == '__main__':
    if '--calibrar-midas' in sys.argv:
        try:
            calibrar_midas()
        except KeyboardInterrupt:
            pass
        sys.exit(0)

    if '--calibrar' in sys.argv:
        # No vuela: solo mide el montaje de la cámara con el dron en el suelo.
        try:
            calibrar_montaje()
        except KeyboardInterrupt:
            pass
        sys.exit(0)

    banner()
    try:
        P.main()
    except KeyboardInterrupt:
        P.activar_paro_emergencia('ctrl-c')
        sys.exit(1)
