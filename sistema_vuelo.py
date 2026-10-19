"""
Dron Crazyflie — Planificador 2D + Ejecución Física con OptiTrack
==================================================================
Versión con ORIENTACIÓN (YAW) TANGENTE A LA RUTA:
el dron siempre apunta hacia donde vuela, girando de forma controlada.

Cambios respecto a la versión anterior:
  · Bloque de parámetros de YAW.
  · Funciones: _wrap_pi, preparar_waypoints, calcular_yaws, planificar_yaw.
  · `ejecutar_ruta_fisica` planifica yaw junto con la posición.
  · Previsualización del rumbo en la ventana de simulación.
  · (Opcional) envío de la orientación del MoCap al EKF.

NUEVO — CÁMARA AI-DECK + MiDaS (integración de `Ai deck MiDas stream.py`)
─────────────────────────────────────────────────────────────────────────
La lógica de ruta NO se tocó: es exactamente la misma. La cámara se añadió
como un subsistema PARALELO y AISLADO que solo mira; nunca manda comandos
de vuelo (excepto el paro de emergencia).

REGLA DE ORO — EL PARO DE EMERGENCIA SIEMPRE FUNCIONA:
  · 'q' en la CONSOLA            → paro (como siempre, vía msvcrt).
  · 'q' o ESC en la VENTANA
    de la cámara                 → el MISMO paro.
    Esto es lo importante: cuando la ventana de OpenCV tiene el foco,
    msvcrt NO ve las teclas de la consola. Por eso la ventana escucha
    también, y ambas llaman a `activar_paro_emergencia()`.
  · La ventana se refresca aunque el stream de vídeo se caiga o se congele,
    así que la tecla responde incluso sin imágenes.
  · MiDaS corre en su PROPIO hilo, con los hilos de torch limitados y con
    prioridad de CPU baja, para no robarle tiempo al hilo del MoCap ni al
    del teclado. Si MiDaS se atrasa, se saltan frames: nunca se bloquea.
  · Cualquier fallo de cámara o de modelo se traga con try/except: el vuelo
    continúa sin visión, nunca al revés.
"""

import json
import sys
import time
import threading
import queue
import socket
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import msvcrt
import os
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────
# ORDEN DE IMPORTS EN WINDOWS: torch ANTES que cv2 y que numpy/mkl
# ─────────────────────────────────────────────────────────────────
# En Windows, si `cv2` (o matplotlib) carga primero su propio libiomp/MKL,
# `import torch` puede reventar con:
#     [WinError 1114] Error en una rutina de inicializacion de DLL.
#     Error loading "...\torch\lib\c10.dll" or one of its dependencies.
# Importar torch el PRIMERO evita ese choque. Además se importa aquí, en el
# hilo principal, y no dentro del hilo de visión: cargar DLLs desde un hilo
# secundario es otra fuente del mismo error.
_PREIMPORTAR_TORCH = True     # ponlo en False si no vas a usar MiDaS nunca

torch = None
_HAY_TORCH = False
_ERROR_TORCH = None
if _PREIMPORTAR_TORCH:
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # choque de OpenMP
    try:
        import torch
        _HAY_TORCH = True
    except Exception as _e_torch:
        _ERROR_TORCH = _e_torch
        torch = None

# OpenCV es opcional: si no está instalado, el vuelo funciona igual, solo
# que sin cámara. Nunca debe impedir despegar.
try:
    import cv2
    _HAY_CV2 = True
except Exception as _e_cv2:
    cv2 = None
    _HAY_CV2 = False
    print(f'⚠ OpenCV no disponible ({_e_cv2}) — se volará sin cámara.')


def _explicar_error_torch(e):
    """Imprime el diagnóstico de por qué torch no carga y cómo arreglarlo."""
    print(f'⚠ PyTorch no se pudo cargar: {e}')
    txt = str(e)
    if '1114' in txt or 'c10.dll' in txt or 'DLL' in txt.upper():
        print('  Es el fallo clásico de torch en Windows. Prueba EN ESTE ORDEN:')
        print('   1) Instala "Microsoft Visual C++ Redistributable x64"')
        print('      (vc_redist.x64.exe de microsoft.com). Es la causa nº1.')
        print('   2) Reinstala torch CPU limpio:')
        print('        pip uninstall -y torch torchvision torchaudio')
        print('        pip cache purge')
        print('        pip install torch torchvision --index-url '
              'https://download.pytorch.org/whl/cpu')
        print('   3) Si tienes numpy 2.x con un torch viejo:  pip install "numpy<2"')
        print('   4) Comprueba que sea Python 64-bit:')
        print('        python -c "import platform;print(platform.architecture())"')
    print('  Mientras tanto se vuela con cámara pero SIN mapa de profundidad.')

from shapely.geometry import Polygon, LineString
from scipy.interpolate import splprep, splev

import paho.mqtt.client as mqtt
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from datetime import datetime

try:
    import trayectoria_suave
    _HAY_TRAYECTORIA = True
except Exception as _e_tray:
    trayectoria_suave = None
    _HAY_TRAYECTORIA = False
    print(f'⚠ trayectoria_suave.py no disponible ({_e_tray}); '
          'se volará con go_to.')

from crazyflie_python_commands_mod import (
    connect, disconnect, takeoff, land,
    move_to_position, set_position
)

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE RED Y DRON
# ─────────────────────────────────────────────
URI          = "radio://0/80/2M/E7E7E7E7F9"
MQTT_BROKER  = '192.168.50.200'
MQTT_PORT    = 1880
MQTT_TOPIC   = 'mocap/drone1'

# ─────────────────────────────────────────────
# PARÁMETROS DE VUELO
# ─────────────────────────────────────────────
HOVER_HEIGHT   = 0.3
FLY_VELOCITY   = 0.15   # m/s REALES sobre el suelo. Rango sensato 0.08-0.25.
                        # El Crazyflie aguanta mucho más; el límite aquí es que
                        # la nariz alcance a girar y que la cámara vea nítido.
VEL_APROX      = 0.15   # m/s para ir del despegue al primer waypoint
TAKEOFF_HEIGHT = 0.3
TAKEOFF_DUR    = 2.5
LAND_HEIGHT    = 0.0
LAND_DUR       = 2.0
DWELL_TIME     = 0.16

MIN_DIST     = 0.12   # distancia mínima entre waypoints (m)
                      # Es lo que fija cuánto se parece la poligonal de
                      # waypoints a la spline real. 0.20 dejaba hasta 6.8 cm
                      # de error de cuerda en las curvas; 0.12 lo baja a 4 cm
                      # y casi no cuesta tiempo.

# SOLAPAMIENTO ya NO cambia la velocidad, solo cuánto se retrasa el dron
# respecto a su consigna. Al mandar el siguiente go_to, el dron solo ha
# recorrido (1-SOLAPAMIENTO) del tramo actual, así que vuela permanentemente
# SOLAPAMIENTO × MIN_DIST por detrás → recorta las curvas por dentro.
#   0.40 con MIN_DIST 0.20 = 8 cm de retraso
#   0.30 con MIN_DIST 0.12 = 3.6 cm
# Bajarlo aprieta la trayectoria; subirlo suaviza pero recorta más.
#
# ── POR QUÉ 0.50 Y NO 0.30 (aquí estaba la frenada) ──
# El firmware planifica cada go_to con un polinomio de 7º grado con
# velocidad, aceleración y jerk nulos en LOS DOS extremos (planner.c,
# compute_7th_order). Su perfil de velocidad, con tau = t/duración, es
#
#       v(tau) ∝ tau³ · (1-tau)³        → pico exacto en tau = 0.50
#
# El siguiente go_to se manda en tau = 1 - SOLAPAMIENTO. O sea que
# SOLAPAMIENTO no decide "cuánto se solapa": decide EN QUÉ PUNTO DE LA
# CURVA DE VELOCIDAD se reencamina el dron.
#
#   SOLAPAMIENTO   tau     velocidad al reencaminar
#       0.20       0.80          26 %
#       0.30       0.70          59 %   ← lo que había: ya frenando
#       0.40       0.60          88 %
#       0.50       0.50         100 %   ← el pico
#
# Con 0.30 el dron se reencamina cuando ya cayó al 59 % de su punta. Esa
# es la desaceleración de cada waypoint: no la causa go_to, la causa
# cortarlo tarde. En 0.50 se corta justo en el pico y el dron no llega
# nunca a entrar en la fase de frenado.
#
# Y hay una segunda razón para 0.50: es el MÁXIMO de la curva, donde la
# derivada es cero. El jitter del sleep de Windows (±15 ms) apenas cambia
# la velocidad ahí. En 0.40, en plena pendiente, ese mismo jitter sí se
# nota. 0.50 es a la vez el más rápido y el más repetible.
#
# Precio: el retraso sube de 3.6 a 6.0 cm (SOLAPAMIENTO × MIN_DIST). Si
# molesta, se compensa bajando MIN_DIST a 0.08 → 4 cm, pero cambia una
# cosa por vuelo: primero esto.
SOLAPAMIENTO = 0.50

# ─────────────────────────────────────────────
# CÓMO SE EJECUTA LA RUTA
# ─────────────────────────────────────────────
#   'trayectoria' : se genera un polinomio por tramos, se sube al dron y el
#                   firmware lo vuela a 500 Hz. Velocidad y aceleración
#                   CONTINUAS: no frena en los waypoints. Además, una vez
#                   subida no hace falta ni un paquete de radio durante el
#                   recorrido, así que el jitter de red deja de afectar.
#   'go_to'       : el método anterior, un go_to por waypoint. Cada go_to
#                   planifica llegar con velocidad CERO, PERO el original
#                   nunca deja que termine: manda el siguiente en
#                   tau = 1-SOLAPAMIENTO. Con SOLAPAMIENTO bien puesto el
#                   dron no frena nunca. ← es el que sigue bien la ruta.
#
# VUELVE A 'go_to'. Motivo, y es una corrección a lo que yo dije antes:
# la trayectoria polinómica es más suave sobre el papel, pero es ABIERTA
# EN EL TIEMPO. El polinomio dice "en t=32.4 s estás en P" y sigue
# avanzando le siga el dron o no; si se retrasa, nadie le espera y el
# error se acumula durante toda la ruta. go_to es CERRADO en posición:
# cada consigna es un punto absoluto y el dron converge a él. Por eso tu
# gráfica del original sigue el rectángulo y la de la trayectoria no.
# Yo cambié esto para quitar las frenadas sin darme cuenta de que el
# original ya tenía su propio mecanismo anti-frenada (SOLAPAMIENTO), solo
# que mal ajustado. Arreglar ese número conserva la precisión y quita las
# frenadas; cambiar de método sacrificaba la precisión para lograr lo
# mismo.
#
#   'pursuit'     : Pure Pursuit con el lazo de posición CERRADO EN EL PC
#                   contra el MoCap, comandando velocidad de mundo a 50 Hz.
#                   ← el de la carpeta `nuevo metodo`, y el que gana.
#
# POR QUÉ 'pursuit' Y NO LOS OTROS DOS. Es una diferencia estructural, no
# de ajuste, y explica un factor 50 en el error medido:
#
#   go_to y trayectoria → el lazo de posición se cierra DENTRO del dron,
#       contra su EKF. Medido en este montaje: 13.9 cm RMS de error de
#       estimación con picos de 60 cm. El dron va exactamente a donde CREE
#       que está el punto, y cree mal. Ningún ajuste arregla eso, porque el
#       error entra por debajo del lazo.
#
#   pursuit → el lazo de posición se cierra EN EL PC, contra la posición
#       del MoCap directamente. El error del EKF no entra en el lazo de
#       posición; el EKF solo lo usa el controlador interno de velocidad y
#       actitud del dron, que es mucho menos sensible.
#
# Medido en este mismo laboratorio con este método (nuevo metodo/.../
# vuelos/metricas_vuelo_real.txt): error lateral MEDIO 1.2 cm, máximo
# 3.2 cm, a 0.30 m/s, en una ruta de radio mínimo 0.24 m y con la batería
# ya a 3.40 V. Contra los 57 cm del último vuelo con trayectoria.
MODO_EJECUCION = 'pursuit'

# ─────────────────────────────────────────────
# PURE PURSUIT  (MODO_EJECUCION = 'pursuit')
# ─────────────────────────────────────────────
PP_CONTROL_HZ  = 50.0    # cadencia del lazo en el PC
PP_V_MAX       = 0.30    # m/s en recta
PP_V_MIN       = 0.10    # m/s mínimo (en la curva más cerrada)
PP_A_LAT_MAX   = 0.35    # m/s² lateral: manda el radio mínimo, v=sqrt(a*r)
PP_A_TAN_MAX   = 0.40    # m/s² tangencial: frenado anticipado y arranque
PP_VZ_MAX      = 0.30    # m/s vertical

# L = PP_LOOKAHEAD_T * v. Debe aproximar el RETARDO del lazo de velocidad
# del Crazyflie (~0.3-0.4 s) para que la tangente de anticipación
# corresponda al punto donde el dron estará cuando el comando surta efecto.
PP_LOOKAHEAD_T = 0.40
PP_L_MIN       = 0.10
PP_L_MAX       = 0.45

PP_KP_LAT      = 1.8     # 1/s. Subir si corta las curvas por dentro;
                         # bajar si oscila alrededor de la ruta.
PP_KP_Z        = 1.3     # 1/s. No subirlo para "suavizar": lo que suaviza
                         # es BAJAR ganancia, no meter filtro (un filtro
                         # dentro del lazo cerrado quita margen de fase).
PP_ALPHA_V     = 0.30    # suavizado de la rapidez objetivo
PP_ALPHA_VZ    = 0.27    # suavizado del comando vertical
PP_V_XY_MAX    = 0.85    # m/s tope del comando horizontal
PP_VZ_MAX_CMD  = 0.45    # m/s tope del comando vertical
PP_TOL_META    = 0.10    # m de arco restante para dar la ruta por hecha
PP_PASO_RUTA   = 0.02    # m de espaciado al remuestrear la ruta

# Cada cuánto se reenvía la pose al EKF del dron. NO es la cadencia del
# lazo: el lazo usa mocap_pose, que se actualiza con cada mensaje. Esto
# solo limita los paquetes de radio, que compiten con los setpoints.
PP_EXTPOS_HZ   = 50.0

# El dron es holónomo y se comanda en velocidad de MUNDO: el yaw no decide
# hacia dónde va, solo hacia dónde mira. Girar mete perturbaciones y
# acelera la deriva del yaw del EKF.
#   'hold' → no gira. Máxima precisión. Es el que usaron en sus 15 vuelos.
#   'path' → la nariz sigue la tangente. Lo que necesita la cámara para
#            mirar hacia donde avanza; cuesta algo de precisión.
PP_YAW_MODE    = 'hold'
PP_YAW_OFFSET  = 180.0   # ° entre el eje X del MoCap y el frente del dron
PP_KP_YAW      = 0.6
PP_YAW_RATE_MAX = 0.5    # rad/s
PP_YAW_ALPHA_REF = 0.08
PP_YAW_ALPHA_OUT = 0.12
PP_YAW_DEADBAND  = 5.0   # °

# Perro guardián: si la posición del MoCap lleva más de esto sin
# refrescarse, se manda velocidad CERO. Con `send_velocity_world_setpoint`
# el dron mantiene el último comando hasta que llegue otro, así que un
# atasco del lazo (la cámara y MiDaS compiten por el GIL) lo dejaría
# volando recto a ciegas. Esto es lo único que no tenía el original, y
# aquí hace falta porque aquí sí hay cámara.
PP_MOCAP_TIMEOUT = 0.5   # s (Robotat a 30 Hz desde oct-2026)
PP_TIMEOUT_FACTOR = 2.5  # aborta si tarda más de X veces lo estimado

# Parámetros de la trayectoria continua.
# El mando principal es TRAY_A_LAT_MAX: cuánta aceleración lateral aceptas
# en las curvas. Es lo que decide si el dron las toma rápido o las saborea.
TRAY_V_CRUCERO   = 0.10    # m/s en recta
TRAY_A_LAT_MAX   = 0.4    # m/s² en curva  (↑ más rápido, ↓ más suave)
TRAY_A_LON_MAX   = 0.35    # m/s² acelerando y frenando
TRAY_V_MIN       = 0.10    # m/s mínimos: nunca se arrastra
TRAY_YAW_RATE    = np.radians(60.0)
TRAY_DESV_OBJETIVO_CM = 3.0   # desviación máxima aceptada frente a la ruta
TRAY_ID          = 1       # identificador en la memoria del dron

# ─────────────────────────────────────────────
# PARÁMETROS DE ORIENTACIÓN (YAW)
# ─────────────────────────────────────────────
YAW_ACTIVO      = True                 # False → vuela como antes (yaw fijo 0)
YAW_RATE_MAX    = np.radians(50.0)     # velocidad angular máx. REAL (rad/s)
                                       # Es un TECHO, no un objetivo: si un tramo
                                       # exigiera más, el planificador lo alarga
                                       # solo. Súbelo si quieres más velocidad.
YAW_UMBRAL_ROT  = np.radians(55.0)     # giro mayor a esto → rota en el sitio y luego avanza
YAW_LOOKAHEAD   = 3                    # waypoints hacia adelante para la tangente
                                       #   1 = la nariz sigue el tramo que está recorriendo
                                       #   2-3 solo si ves que apunta hacia ATRÁS
YAW_VENTANA     = 3                    # ventana (impar) de suavizado del ángulo
                                       #   ↑ suaviza, pero desalinea la nariz en curvas cerradas
YAW_OFFSET      = 0.0                  # rad. Ángulo de montaje de la cámara
YAW_PAUSA_ROT   = 0.4                  # s. Estabilización tras girar en el sitio
YAW_T_MIN_ROT   = 1.0                  # s. Duración mínima de una rotación en el sitio

# En el giro más cerrado el dron vuela a este % de FLY_VELOCITY (0.6 = 60%).
# Bajarlo da más tiempo a la nariz para alinearse; subirlo acorta el vuelo.
VEL_FACTOR_GIRO = 0.6

# Enviar la posición al EKF con cflib directamente, sin pasar por set_position.
# Evita el print "Absolute position successfully set." en cada mensaje del
# MoCap, que satura la consola y retrasa las correcciones del estimador.
USAR_EXTPOS_DIRECTO = True
MOSTRAR_HZ_MOCAP    = True   # imprime la frecuencia real del MoCap cada 5 s

# Enviar también la ORIENTACIÓN del MoCap al EKF.
#
# Esto pasó de "opcional" a ser probablemente LO MÁS IMPORTANTE para la
# precisión. Con solo posición, el yaw del EKF sale de integrar el
# giroscopio y deriva. El controlador usa ese yaw para convertir el error de
# posición (en coordenadas del mundo) en roll/pitch (en coordenadas del
# cuerpo): si el yaw está desfasado 30°, el dron empuja 30° torcido y la
# trayectoria entera sale girada y agrandada respecto a la planificada.
#
# Con yaw fijo casi no se nota. Con yaw tangente —que es tu caso, y la ruta
# gira más de 360° en total— la deriva se acumula durante todo el vuelo.
#
# Antes de ponerlo en True, comprueba con `python diagnostico.py mocap` que
# el mensaje trae el cuaternión y que está normalizado.
USAR_ORIENTACION_MOCAP = False

# Incertidumbre que el EKF asigna al cuaternión externo. Más bajo = más se
# fía del MoCap. Recomendado por Bitcraze para MoCap: 0.05.
EXT_QUAT_STD_DEV = 0.05

# ─────────────────────────────────────────────
# CUÁNTO SE FÍA EL EKF DE LA POSICIÓN EXTERNA
# ─────────────────────────────────────────────
# Este parámetro es la pieza que faltaba, y explica por qué subir la cadencia
# a 10 Hz empeoró las cosas en vez de mejorarlas.
#
# El EKF trata cada `extpos` como una medida con esta desviación típica. Si
# la considera muy fiable (valor bajo), SALTA a ella. A 1 Hz eso es un tirón
# por segundo; a 10 Hz son diez tirones por segundo, y como el dato llega con
# retraso —Motive → Node-RED → MQTT → Python → radio— el dron corrige hacia
# donde ESTABA hace un rato. Ese desfase realimentado diez veces por segundo
# es precisamente una oscilación: es el mecanismo clásico de inestabilidad
# por retardo, y por eso se disparó en vertical.
#
# La solución NO es volver a 1 Hz ni subir más los Hz: es decirle al EKF que
# la medida es buena pero no perfecta, para que la FILTRE en vez de saltar a
# ella. Muchas medidas poco pesadas dan una estimación suave; pocas medidas
# muy pesadas dan escalones.
#
#   0.01  muy fiable → salta (por defecto suele rondar esto)
#   0.05  compromiso razonable con MoCap a 10-30 Hz
#   0.10  filtra bastante; úsalo si aún oscila
#
# None = no tocar el parámetro (deja el que traiga el firmware).
EXT_POS_STD_DEV = 0.05
CONTROLADOR_ESTABILIZADOR = None   # None = no tocar (el pursuit, que despega bien, no lo toca); 1 = PID

# ─────────────────────────────────────────────
# FILTRO DE TIMESTAMP DEL MOCAP
# ─────────────────────────────────────────────
# El filtro original exigía que cada `ts` fuese ESTRICTAMENTE mayor que el
# anterior, para descartar mensajes repetidos o desordenados. El problema:
# el publicador manda el `ts` con resolución de UN SEGUNDO. Llegan ~85
# mensajes por segundo, todos con el mismo `ts`, y el filtro tiraba 84.
# Medido con diagnostico.py: 1702 recibidos (85.1 Hz) → 21 aceptados (1.1 Hz).
#
#   'estricto' : comportamiento original (ts > anterior). Solo sirve si el
#                publicador manda milisegundos.
#   'tolerante': acepta ts IGUAL al anterior y rechaza los más viejos. Con
#                resolución de segundo es lo máximo que se puede filtrar,
#                y deja pasar todo el flujo. ← el que arregla tu caso.
#   'contenido': NO mira el `ts`. Deduplica comparando la POSICIÓN: si
#                cambió, es muestra nueva; si es idéntica, es la misma
#                repetida. Es lo que el filtro quería hacer desde el
#                principio, y no depende de la resolución de la fecha.
#                ← el bueno mientras Node-RED no mande milisegundos.
#   'ninguno'  : ignora el `ts` y acepta todo por orden de llegada.
MODO_FILTRO_TS = 'tolerante'

# Movimiento mínimo (m) para considerar que una muestra es NUEVA en el modo
# 'contenido'. El MoCap tiene ruido de ~0.1 mm; 0.2 mm deja pasar todo
# movimiento real y descarta la muestra repetida byte a byte.
MOCAP_MIN_MOV = 0.0002

# CÓMO se manda la pose al EKF:
#   'callback' : se manda dentro del callback de MQTT, con limitador. ←
#                recomendado, y medido: `send_extpos` tarda 0.0 ms, así que
#                no frena nada. Como el callback ya se ejecuta a ~85 Hz, el
#                limitador dispara en cuanto pasan los ms pedidos: no hay
#                sleep, ni granularidad de reloj, ni un hilo más peleando
#                por el GIL. Es el que da la cadencia real más alta.
#   'hilo'     : un hilo aparte manda a ritmo fijo. Suena mejor sobre el
#                papel, pero en Windows y con MiDaS ocupando la CPU se
#                queda en ~22 Hz aunque le pidas 50: entre la granularidad
#                del temporizador y la contienda por el GIL, el hilo no
#                despierta a tiempo. Déjalo solo si tu MQTT llega a menos
#                de MOCAP_HZ_MAX y quieres repetir la última pose.
MOCAP_MODO_ENVIO = 'callback'

# Tope de correcciones por segundo que se mandan al EKF.
# No hace falta mandar los 85 Hz: por encima de ~50 Hz no mejora el
# estimador y sí ocupa tiempo de aire de la radio, que compite con los
# go_to de la ruta. 0 = sin límite.
MOCAP_HZ_MAX = 50.0

# ─────────────────────────────────────────────
# COLOR LED DECK (LED inferior)
# ─────────────────────────────────────────────
# Se enciende al llegar al primer waypoint y se apaga al terminar la ruta,
# antes de aterrizar. Formato wrgb8888 = 0xWWRRGGBB (1 byte por canal).
LED_ACTIVO  = False
LED_GRUPO   = 'colorLedBot'
LED_BRILLO  = 255            # colorLedBot.brightCorr (0-255)
LED_COLOR   = 0x000000FF     # azul
# Otros colores útiles:
#   rojo    0x00FF0000     verde   0x0000FF00     azul    0x000000FF
#   amarillo 0x00FFFF00    cian    0x0000FFFF     blanco  0xFF000000

# ─────────────────────────────────────────────
# CÁMARA AI-DECK 1.1 + MiDaS
# ─────────────────────────────────────────────
# Todo este subsistema es OPCIONAL y AISLADO. Si algo falla (no hay WiFi del
# deck, no hay torch, no hay OpenCV), se avisa y el vuelo sigue igual.
VISION_ACTIVA    = True          # False → vuela exactamente como antes
VISION_IP        = '192.168.4.1' # IP del AI-deck (AP propio del deck)
VISION_PUERTO    = 5000
VISION_JPEG_BAYER_GRIS = True  # El JPEG del GAP8 es el mosaico Bayer comprimido;
                               # True = se reconstruye y se pasa a gris para quitar el ajedrez.
VISION_MONO      = False         # True si tienes la cámara monocromática
VISION_ESCALA    = 2             # zoom de la ventana (la cámara es 324x244)

# ─────────────────────────────────────────────
# AJUSTES DE IMAGEN Y COLOR
# ─────────────────────────────────────────────
# Estos son EXACTAMENTE los mismos parámetros que el banco de pruebas
# (`depth_simulator/Ai deck MiDas stream.py`), con los mismos nombres y las
# mismas escalas, para que lo que ajustes allí valga aquí sin traducir nada.
#
# Los valores de abajo son los que sacaste en el banco el 2026-08-22.
VISION_AJUSTES = {
    # ── Tono / luz ──
    'realce'     : 2,     # 0 nada · 1 auto · 2 CLAHE · 3 gamma
    'clahe_clip' : 30,    # /10  → 3.0
    'clahe_tile' : 5,
    'gamma'      : 100,   # /100 → 1.0   (<100 aclara, >100 oscurece)
    'brillo'     : 37,    # desplazamiento lineal, -100..100
    'contraste'  : 140,   # /100 → 1.4
    'promedio'   : 1,     # frames a promediar. EN VUELO DEJAR EN 1:
                          # promediar con la cámara en movimiento emborrona.
    'denoise'    : 3,
    'enfoque'    : 0,
    # ── Color ──
    'modo_color' : 0,     # 0 color · 1 gris · 2/3/4 canal R/G/B ·
                          # 5 ExG · 6 VARI · 7 falso color
    'wb'         : 1,     # 0 off · 1 gray-world · 2 white-patch · 3 manual
    'gan_r'      : 99,    # ganancias del balance manual (wb = 3)
    'gan_b'      : 100,
    'saturacion' : 100,   # %
    'demosaico'  : 0,     # 0 rápido · 1 VNG · 2 EA
    'colormap'   : 0,     # paleta para índices y falso color
    'patron'     : 'BG',  # patrón Bayer: BG / GB / RG / GR
    # ── Orientación (montaje físico de la cámara) ──
    # Si la imagen sale de cabeza, aquí se arregla. Ajústalo en el banco
    # con las teclas 't' y 'y', que se ve al momento.
    'rotacion'   : 0,     # 0=0° · 1=90° · 2=180° · 3=270°
    'espejo'     : 0,     # 0=no · 1=horizontal · 2=vertical
}

# JSON que escribe el banco de pruebas al salir. Si aparece, SUS valores
# pisan a los de arriba. Así no hay que copiar nada a mano ni arriesgarse a
# transcribir mal un número: ajustas en el banco, sales, y el vuelo ya usa
# esos ajustes.
#
# Se buscan varias rutas porque el banco escribe en el directorio DESDE EL
# QUE LO LANZAS, no donde está el archivo: si lo ejecutas desde la raíz del
# proyecto, la carpeta acaba en la raíz. Se usa la primera que exista.
# Pon la lista vacía para ignorarlo y usar siempre VISION_AJUSTES.
VISION_AJUSTES_JSON = [
    './pruebas_camara/ultimos_ajustes.json',
    './depth_simulator/pruebas_camara/ultimos_ajustes.json',
]

# MiDaS
VISION_MIDAS     = True          # False → solo vídeo, sin profundidad
VISION_MODELO    = 'MiDaS_small' # MiDaS_small / DPT_Hybrid / DPT_Large
# Lado de entrada para Depth Anything; None = el del procesador (518).
# Medido: 336 detecta mejor y es ~3x más rápido con la cámara del AI-deck.
VISION_DA_LADO = 336
VISION_DISPOSITIVO = 'auto'      # auto / cpu / cuda / mps

# ── Presupuesto de CPU (esto es lo que protege el paro de emergencia) ──
# MiDaS en CPU puede comerse todos los núcleos y dejar sin tiempo al hilo
# del MoCap (el EKF se queda ciego) y al del teclado (el paro tarda en
# responder). Estos dos límites lo evitan:
# Núcleos para torch. 0 = automático (la mitad de los lógicos, que es lo que
# midió mejor: en una máquina de 24 núcleos, 12 hilos dan 38 ms por
# inferencia; con 24 sube a 46 por la contienda, y con 2 se va a 102).
# Ponlo a un número concreto solo si quieres reservar CPU para otra cosa.
VISION_HILOS_TORCH = 0
# Segundos mínimos entre inferencias. Con MiDaS a ~40 ms y el enlace a ~7 fps
# (135 ms por imagen), puede correr en TODAS: el límite ya no hace falta.
VISION_MIN_PERIODO = 0.05
VISION_PRIORIDAD_BAJA = True     # baja la prioridad del hilo de visión (Windows)

# ── Reconexión automática del stream ──
# El WiFi del AI-deck (ESP32/NINA) es notoriamente frágil bajo carga: se cae
# solo, sobre todo en modo AP y con imágenes RAW. Antes, una caída dejaba la
# cámara muerta el resto del vuelo. Ahora el receptor reintenta solo.
# Watchdog del stream: si pasan estos segundos sin recibir un frame
# completo, se da el enlace por atascado y se reconecta. El GAP8 puede
# quedarse colgado dejando el socket TCP abierto: sin esto, la imagen se
# congela para siempre y ningún contador se entera.
VISION_TIMEOUT_LECTURA = 4.0
# El socket del AI-deck se lee en un PROCESO aparte (puente_camara.py), con su
# propio GIL. En el mismo proceso que la radio, el MoCap y el control, el
# receptor esperaba turno en cada una de sus ~160 lecturas por imagen: medido
# sin hardware, con 8 hilos ocupados llegaban 0.7 fps de 3.1 y con hasta 4 s
# de retraso. False = receptor en un hilo de este proceso, como antes.
VISION_RECEPTOR_PROCESO = True

VISION_RECONECTAR   = True
VISION_ESPERA_RECON = 2.0        # s entre reintentos
VISION_MAX_REINTENTOS = 0        # 0 = reintentar indefinidamente
# Reconexiones seguidas sin recibir NI UN frame antes de dar el GAP8 por
# colgado y dejar de insistir. Es un estado del que no se sale por software:
# hay que apagar y encender el dron.
VISION_MAX_SESIONES_VACIAS = 3

# ── Vigilante de CAUDAL (añadido 17-sep-2026, tras el banco de cámara) ──
# VISION_TIMEOUT_LECTURA solo caza un GAP8 MUDO. El fallo del vuelo del
# 16-sep fue otro: el GAP8 seguía mandando, pero a 0.76 fps. El socket nunca
# pasaba 4 s sin datos, así que no saltaba ningún watchdog, no subía ningún
# contador, y desde la PC se veía como "la cámara va lenta" en vez de "la
# cámara está averiada". Se perdió un vuelo entero averiguándolo.
#
# Los números de referencia salen de `probar_camara.py` con este mismo dron:
#
#     enlace pelado ................. 6.6 fps   (~510 kB/s, RAW 324x244)
#     + demosaico (0.1 ms) .......... 6.6 fps
#     + MiDaS (35 ms) ............... 6.9 fps
#     + ventana (47 ms) ............. 6.9 fps
#     + hilos de vuelo .............. 7.0 fps   ← la PC no cuesta NADA
#     + radio abierta ............... 5.2 fps   ← -26 %, y es físico: la
#                                                 Crazyradio y el AI-deck
#                                                 comparten los 2.4 GHz a
#                                                 centímetros el uno del otro
#
# O sea: en vuelo lo normal son ~5 fps. Por debajo de VISION_FPS_MINIMO no
# hay ninguna explicación buena, así que se reconecta en vez de volar a
# ciegas creyendo que se ve.
VISION_FPS_MINIMO = 2.0          # 0 = no vigilar el caudal
VISION_VIGILANTE_GRACIA_S = 8.0  # ventana de medición (y margen de arranque)
# Reconectar no cura un GAP8 medio colgado. Tras estos intentos se deja de
# vigilar —para no entrar en un bucle de reconexiones— y se dice en claro que
# hay que quitar y poner la batería.
VISION_VIGILANTE_MAX = 3

# ── Monitor de batería ──
# Si el SSID del deck DESAPARECE y REAPARECE, el ESP32 se está reiniciando.
# La causa nº1 es un brownout: al volar, los motores hunden la tensión de la
# batería, y si encima el ESP32 está transmitiendo (pico de corriente), el
# regulador se queda sin margen y el ESP32 se resetea.
# Esto registra pm.vbat y lo cruza con cada caída del WiFi: si la tensión se
# hunde justo en cada caída, ya no hay que adivinar más.
# ─────────────────────────────────────────────
# MONITOR DEL ESTIMADOR  (¿el dron no sabe dónde está, o no obedece?)
# ─────────────────────────────────────────────
# Registra lo que el EKF del dron CREE que es su posición y lo compara con
# lo que dice el MoCap. Eso parte el error total en dos causas que hay que
# atacar de forma distinta:
#
#   · error de ESTIMACIÓN = |EKF − MoCap|
#       El dron vuela bien... hacia donde cree que está. Se arregla dándole
#       más correcciones de posición (más Hz de MoCap), no bajando la
#       velocidad ni tocando el controlador.
#   · error de CONTROL    = |EKF − trayectoria comandada|
#       El dron sabe dónde está pero no consigue seguir la referencia. Eso
#       sí se arregla bajando la velocidad o suavizando la trayectoria.
#
# Sin esta separación es imposible saber cuál de las dos cosas arreglar.
MONITOR_ESTIMADOR = True
ESTIMADOR_PERIODO_MS = 100

MONITOR_BATERIA   = True
# Avisos en vivo de tensión. Una LiPo 1S del Crazyflie va de 4.2 V (llena) a
# ~3.0 V (vacía). Por debajo de 3.0 V en vuelo la estás dañando, y el ESP32
# del AI-deck empieza a resetearse por falta de margen.
BATERIA_AVISO_V   = 3.30   # primer aviso
BATERIA_CRITICA_V = 3.00   # aviso fuerte y repetido

# BARRERA DE DESPEGUE. Se mide la tensión ANTES de armar y, si no llega,
# no se vuela. Una LiPo 1S cargada marca 4.0-4.2 V en reposo; por debajo de
# 3.70 V le queda poca carga y en cuanto los motores tiren se hunde a 2.x,
# que es donde el dron no sostiene actitud y el GAP8 se cuelga.
# Existe porque cuatro vuelos seguidos se hicieron con la batería agotada
# sin que nada lo impidiera: los avisos por consola se perdían entre el
# resto de la salida.
BATERIA_MINIMA_DESPEGUE = 3.70
PERMITIR_VOLAR_SIN_BATERIA = False   # True = solo avisar, no bloquear
BATERIA_PERIODO_MS = 100         # 10 Hz. OJO si tu URI va a 1M en vez de 2M:
                                 # a 1M cada paquete ocupa el doble de tiempo
                                 # de aire. Si al activar esto ves bajar el
                                 # "[MoCap] xx Hz", sube a 200 (5 Hz) o pon
                                 # MONITOR_BATERIA = False.
BATERIA_VENTANA_S  = 3.0         # s hacia atrás que se miran en cada caída

# Posición de la LAPTOP (o del router, si el deck va en modo estación) en
# coordenadas del MoCap, en metros. Sirve para el informe de caídas: permite
# comprobar si el enlace se cae MÁS LEJOS o da igual dónde esté el dron.
# Déjalo en None si no la sabes; entonces se mide contra el origen.
VISION_POS_LAPTOP = None         # p. ej. (0.0, -1.5, 0.9)

# Guardado automático de capturas durante el vuelo (0 = desactivado).
# Cada captura guarda RGB + mapa de profundidad + la pose del MoCap.
VISION_GUARDAR_CADA = 0.0        # segundos entre capturas automáticas
VISION_DIR_CAPTURAS = './capturas_vuelo'

# ─────────────────────────────────────────────
# PARÁMETROS DE RUTA 2D
# ─────────────────────────────────────────────
AREA_X   = 2.0
AREA_Y   = 2.5
PASO     = 0.35      # ← SEPARACIÓN ENTRE FRANJAS del barrido, en metros.
                     #   Ver la nota al final de este bloque.
SUAVIDAD = 0.00001
N_SPLINE = 500

# ══════════════════════════════════════════════════════════════
# IMAGEN DE FONDO EN LA PANTALLA DE DIBUJO   (opcional, 18-sep-2026)
# ══════════════════════════════════════════════════════════════
# Pone una foto cenital del cultivo DEBAJO de la cuadrícula donde marcas
# los waypoints, para poder elegir a ojo qué parte quieres regar. Es la
# misma imagen que proyectarías con la cañonera, así que lo que ves en la
# pantalla y lo que hay en el suelo coinciden.
#
# ES OPCIONAL Y ESTÁ APAGADA. Con FONDO_IMAGEN = None todo funciona
# exactamente igual que antes: misma cuadrícula, mismos ejes, mismos
# clics. No toca la generación de la ruta ni el control: es un `imshow`
# por debajo de todo (zorder=0) y nada más.
#
# Para usarla, tres formas (ninguna obligatoria):
#
#   1. La CÓMODA, y la que evita andar renombrando archivos: con la
#      ventana de dibujo abierta, pulsa  'f'  y sale el explorador de
#      Windows para escoger la imagen. Puedes cambiarla las veces que
#      quieras sin cerrar nada, y  'g'  la quita. Los puntos que lleves
#      marcados no se tocan: la imagen es una capa aparte.
#
#   2. Al arrancar, con el explorador:    --fondo
#      Al arrancar, con un archivo fijo:  --fondo cultivo.png
#
#   3. Fija en el archivo, si siempre usas la misma:
#          FONDO_IMAGEN = './cultivo.png'
#
# El rectángulo se define en METROS del mismo sistema del MoCap, así que
# la imagen queda anclada al laboratorio y no a los píxeles.
FONDO_IMAGEN  = None          # ruta a un PNG/JPG, o None para no usarla
FONDO_ANCHO_M = 2.0           # ancho del rectángulo proyectado (eje X)
FONDO_ALTO_M  = 3.0           # alto  del rectángulo proyectado (eje Y)
FONDO_CENTRO  = (0.0, 0.0)    # dónde cae el centro de la imagen (x, y)
FONDO_ROTAR   = 0             # 0, 90, 180 o 270 grados. Al girar 90 o 270
                              # se intercambian ANCHO y ALTO solos, para
                              # que la foto no se deforme. En la ventana,
                              # la tecla 'r' gira 90° cada vez.
FONDO_ALPHA   = 0.55          # 1.0 = opaca; conviene < 1 para ver la rejilla
FONDO_BORDE   = True          # dibujar el contorno del área proyectada
# Si prefieres dar las cuatro esquinas a mano, pon aquí
# (x_min, x_max, y_min, y_max) y se ignoran ANCHO/ALTO/CENTRO.
FONDO_EXTENT  = None

# ── NOTA SOBRE `PASO` (la separación entre franjas) ──
# `PASO` es el que separa las pasadas del barrido en zig-zag. Se usa en
# `generar_ruta(poligono, PASO)`, y SOLO cuando se dibuja un polígono
# nuevo.
#
# OJO: una ruta GUARDADA lleva dentro el PASO con el que se generó. Si la
# cargas desde el menú, cambiar este número no le hace nada — hay que
# volver a dibujar el área para que se aplique. Por eso el arranque
# imprime `Parámetros: PASO=...`: ese es el de la ruta que vas a volar,
# no el de este archivo.

# ─────────────────────────────────────────────
# PARÁMETROS PURE PURSUIT (solo simulación)
# ─────────────────────────────────────────────
LOOKAHEAD   = 0.01
VEL_MAX     = 0.10
VEL_MIN     = 0.01
KP          = 1.0
KD          = 0.3
DT          = 0.05
UMBRAL_FIN  = 0.03
ESCALA_DRON = 0.06

# ─────────────────────────────────────────────
# ARCHIVO DE RUTA
# ─────────────────────────────────────────────
RUTA_DIR  = './rutas_guardadas'
os.makedirs(RUTA_DIR, exist_ok=True)

# Con varios hilos (MQTT a 85 Hz, MiDaS, ventana, vuelo) el intervalo de
# conmutación por defecto de Python (5 ms) hace que un hilo sensible al
# tiempo espere demasiado a que le toque el GIL. Bajarlo a 1 ms reparte
# mejor y le quita latencia al envío de correcciones.
try:
    sys.setswitchinterval(0.001)
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════
# MODO COMPATIBLE CON EL ORIGINAL  ← LEE ESTO ANTES DE TOCAR NADA
# ═══════════════════════════════════════════════════════════════
# Con True, la ruta de datos MoCap → EKF y el tráfico de radio quedan
# EXACTAMENTE como en la primera versión del planificador. Lo único que se añade es la cámara,
# que vive en otra red y no toca la radio.
#
# POR QUÉ EXISTE ESTO
# El original acepta ~1 mensaje por segundo (su filtro de timestamp descarta
# el resto) y vuela bien. Mi versión subió eso a 20-50 Hz y el dron empezó a
# oscilar. Suena al revés, pero tiene explicación: el EKF pondera la posición
# externa según lo seguido que llegue. A 1 Hz la trata como una corrección
# ocasional y navega con la IMU, que no tiene retardo. A 50 Hz la trata como
# su fuente principal — y esa fuente viaja por Motive → Node-RED → WiFi →
# Python → Crazyradio, con un retardo de decenas o cientos de ms. El EKF
# corrige entonces hacia donde el dron ESTABA, el controlador reacciona
# tarde, y ese desfase realimentado es exactamente una oscilación.
#
# ACTUALIZACIÓN TRAS MEDIR (esto de arriba era una TEORÍA, y era falsa)
# La latencia se midió por fin bien, contra la pose sin filtrar: −20 ms.
# La cadena Motive → Node-RED → MQTT → Python va prácticamente al día. Así
# que el párrafo de arriba —"los datos llegan viejos"— no describe este
# montaje. Lo que pasa de verdad, medido en el mismo vuelo:
#
#   · Llegan ~93 mensajes por segundo (8670 muestras en 93 s de vuelo).
#   · El filtro estricto acepta 1.0 Hz, porque `ts` tiene resolución de UN
#     SEGUNDO y todos los mensajes del mismo segundo parecen repetidos.
#   · Entre corrección y corrección el dron recorre 24 cm a ciegas.
#   · El error de estimación es 13.9 cm RMS y llega a 60 cm de pico, y NO
#     es artefacto de la medida: medido contra la pose cruda sale igual.
#
# O sea: el dron no sigue la ruta porque no sabe dónde está, y no lo sabe
# porque estamos tirando el 99 % de un dato que llega perfectamente a
# tiempo. El cuello de botella era el filtro, no la red.
#
# PERO: subir la cadencia es un experimento APARTE, y ahora mismo NO toca.
# Lo puse en 5 Hz y el despegue empeoró, así que vuelve a True. Es lo único
# que había cambiado, así que es lo primero que se revierte: una variable
# por vuelo, o no se aprende nada. La cadencia del MoCap sigue siendo una
# mejora pendiente, pero se prueba DESPUÉS y sola, no mezclada con el
# ajuste de SOLAPAMIENTO.
#
# ESCALERA PARA MÁS ADELANTE (un vuelo por escalón, batería CARGADA):
#   1. MODO_COMPATIBLE_ORIGINAL = False, 'contenido', MOCAP_HZ_MAX = 5
#      → 5 cm a ciegas en vez de 24. Es la mitad de los 10 Hz que oscilaron.
#   2. Si sale limpio: MOCAP_HZ_MAX = 10 → 20 → 30.
#   El primero que oscile marca el techo; quédate en el anterior.
#
# Y si vuelve a oscilar, el sospechoso YA NO es la latencia (está medida):
# es cuánto se fía el EKF de cada corrección. Sube EXT_POS_STD_DEV
# (0.05 → 0.10) antes de bajar la cadencia.
MODO_COMPATIBLE_ORIGINAL = True

if MODO_EJECUCION == 'pursuit':
    # Pure Pursuit necesita lo contrario que los otros dos métodos: la
    # posición LO MÁS FRESCA POSIBLE, porque el lazo de posición corre aquí
    # y se alimenta de `mocap_pose` directamente. Filtrarla lo ciega.
    #
    # Las dos cadencias se separan (es la pieza que me faltaba):
    #   · `mocap_pose` → se actualiza con CADA mensaje (~93 Hz), sin filtro.
    #   · extpos al EKF → limitado a PP_EXTPOS_HZ, porque cada envío es un
    #     paquete de radio que compite con los setpoints de velocidad.
    # Aquí el EKF ya no lleva el lazo de posición: solo sostiene el
    # controlador interno de velocidad y actitud.
    MODO_COMPATIBLE_ORIGINAL = False
    MODO_FILTRO_TS   = 'ninguno'
    MOCAP_MODO_ENVIO = 'callback'
    MOCAP_HZ_MAX     = PP_EXTPOS_HZ
    BATERIA_PERIODO_MS = 500
    print(f'[cfg] PURE PURSUIT — lazo de posición en el PC contra el MoCap '
          f'a {PP_CONTROL_HZ:.0f} Hz.')
    print(f'      mocap_pose sin filtrar (~93 Hz) | extpos al EKF a '
          f'{PP_EXTPOS_HZ:.0f} Hz.')
elif MODO_COMPATIBLE_ORIGINAL:
    MODO_FILTRO_TS   = 'estricto'   # filtro original: ts estrictamente creciente
    MOCAP_MODO_ENVIO = 'callback'   # envío original: dentro del callback
    MOCAP_HZ_MAX     = 0            # sin limitador (el filtro ya limita)
    # El monitor de batería SE QUEDA, pero a 2 Hz en vez de 10.
    # Antes lo apagaba del todo por ahorrar radio, y fue un error: con el
    # EKF a 1 Hz la radio está casi ociosa, así que 2 paquetes por segundo
    # no molestan a nadie. Y volar sin ver la tensión nos dejó ciegos justo
    # ante la causa conocida de que el GAP8 se cuelgue.
    BATERIA_PERIODO_MS = 500
    # El monitor del estimador se queda: es solo un log de lectura y es lo
    # que permite saber si el error es de estimación o de control.
    print('[cfg] MODO_COMPATIBLE_ORIGINAL=True — ruta MoCap→EKF y radio '
          'idénticas a la primera versión. Solo se añade la cámara.')
else:
    # Escalón 1 de la escalera de arriba. 'contenido' no mira el `ts`, así
    # que la resolución de un segundo deja de importar; el limitador de Hz
    # es quien decide la cadencia, y empieza bajo a propósito.
    MODO_FILTRO_TS   = 'contenido'
    MOCAP_MODO_ENVIO = 'callback'
    MOCAP_HZ_MAX     = 5
    print(f'[cfg] MoCap por CONTENIDO a {MOCAP_HZ_MAX} Hz — '
          f'~{0.24/MOCAP_HZ_MAX*100:.0f} cm a ciegas en vez de 24. '
          f'Si oscila, sube EXT_POS_STD_DEV antes de bajar Hz.')

# ── REGISTRO DE TRAYECTORIA A ALTA CADENCIA ──
# Independiente de lo anterior, y esta es la clave para poder MEDIR: separa
# "qué se le manda al dron" de "qué se apunta en el cuaderno".
#
# El filtro de timestamp descarta el 98% de los mensajes ANTES de guardarlos,
# así que en modo compatible la trayectoria real se registra a ~1 Hz: 60
# muestras para un vuelo de 60 s. Con eso el RMS que sale es una estimación
# muy pobre, y es el número que acaba en la tesis.
#
# Con esto en True, TODOS los mensajes del MoCap (~85 Hz) se guardan en
# `real_trajectory` para el análisis posterior, mientras al EKF se le manda
# exactamente lo mismo que antes. El vuelo no cambia en nada: no añade ni un
# paquete de radio. Solo mejora la medición.
REGISTRAR_TRAYECTORIA_COMPLETA = True

# ─────────────────────────────────────────────
# ESTADO GLOBAL
# ─────────────────────────────────────────────
puntos_clicks   = []
mocap_pose      = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}
# Pose SIN filtrar: se actualiza con CADA mensaje que llega del MoCap,
# antes del filtro de timestamp. `mocap_pose` solo cambia cuando el filtro
# deja pasar un mensaje; a 1 Hz eso es una escalera que se queda congelada
# un segundo entero. Medir el error del estimador contra esa escalera no
# mide el estimador: mide la escalera. Esta copia es la referencia buena.
mocap_pose_cruda = {'x': 0.0, 'y': 0.0, 'z': 0.0}
_mocap_seq_cruda = 0
real_trajectory = []

# ─────────────────────────────────────────────
# FILTRO DE SALTOS DEL MOCAP (glitch de un solo cuadro)
# ─────────────────────────────────────────────
# 25-sep-2026: en un despegue el dron saltó 18.7 cm en 21 ms (8.9 m/s,
# físicamente imposible para este dron) EXACTAMENTE en el instante de
# despegar, y ese salto —no la orientación, que se descartó como causa—
# desestabilizó el vuelo: la corrección de posición hace que el EKF vea una
# velocidad falsa y el controlador se inclina para "cancelarla". Pasaba en
# unos puntos del Robotat y no en otros: encaja con cobertura de cámaras
# desigual (borde del volumen, oclusión) que se agrava con la vibración de
# las hélices al arrancar. Esto descarta ESE cuadro (mantiene la muestra
# anterior) en vez de mandarlo al EKF o al lazo de control.
#
# SEGUNDA VUELTA (mismo día): con "tras 3 rechazos, se acepta igual" puesto
# más abajo, otro despegue coló un salto de 72 cm —esta vez el MoCap se
# quedó CONGELADO 3.6 s (mismo valor exacto, no ruido) y al volver llegó de
# golpe una posición muy distinta; los 3 rechazos se gastaron en ese primer
# instante y la 4ª muestra, todavía mala, se aceptó igual porque "ya habían
# sido demasiados". Para ESTE dron ninguna velocidad por encima de
# MOCAP_SALTO_MAX_VEL es real nunca, así que no hay número de rechazos que
# haga razonable aceptarla: se quita esa válvula. Lo del MoCap congelado se
# ataja aparte, con el vigía de frescura ya existente (_t_mocap_fresco /
# MOCAP_TIMEOUT), ahora también antes de despegar.
MOCAP_SALTO_MAX_VEL = 3.0    # m/s: por encima de esto, no es un movimiento
                             # real del dron (su velocidad de crucero es
                             # 0.10-0.35 m/s); se descarta como glitch.
# MQTT entrega en ráfagas: el tiempo entre llegadas no es el de las muestras.
# 3 m/s × 0.04 s admite 12 cm; sigue rechazando saltos reales de 18.7 y 72 cm.
MOCAP_SALTO_DT_MIN = 0.04
MOCAP_SALTO_DT_MAX = 0.1     # s: el "presupuesto" (velocidad × tiempo desde
                             # la última muestra buena) se tapa aquí. Sin
                             # este tope, un valor atascado (congelado o
                             # relocalizado a un punto fantasma, quieto)
                             # acababa pareciendo plausible con solo esperar
                             # lo suficiente: a más segundos, más margen, y
                             # cualquier salto fijo entra tarde o temprano.
                             # Con el tope, ese margen deja de crecer.
MOCAP_SALTO_DURACION_PARO_S = 0.5  # s: rechazando seguido más que esto, algo
                                   # va mal de verdad (no un cuadro suelto):
                                   # mejor parar que seguir volando a ciegas.
_mocap_gate = {'x': None, 'y': None, 'z': None, 't': None,
              't_primer_rechazo': None}

# ─────────────────────────────────────────────
# MOCAP CONGELADO: llegan mensajes, pero con la MISMA posición exacta
# ─────────────────────────────────────────────
# La causa real de "despega yéndose de lado si no lo pongo en el punto
# exacto" (25-sep-2026, 9 registros de despegue): en TODOS los fallidos la
# posición llegaba idéntica bit a bit desde ANTES de encender motores, 1.8 a
# 4.3 s, y al "descongelarse" saltaba 50-60 cm. En los buenos cambia siempre
# (ruido mínimo medido: 0.007 mm). Es Motive que no ve el dron en ese punto
# del suelo, mientras el puente sigue repitiendo la última pose vista (en la
# mano, al colocarlo). El EKF arranca creyendo que el dron está donde no
# está; al subir, las cámaras lo vuelven a ver, la posición salta a la real
# y el controlador lo lanza de lado para "corregir". Como llegan mensajes,
# el vigía de frescura (_t_mocap_fresco) no lo notaba: hace falta mirar si
# la posición CAMBIA, no solo si llega.
MOCAP_CONGELADO_S = 0.5      # s con la posición idéntica = no se está rastreando.
                             # Robotat a 30 Hz desde oct-2026: con 0.25 saltaba en huecos
                             # normales; los congelamientos reales duraban 1.8-4.3 s.
_t_mocap_cambio = [0.0]      # perf_counter del último cambio real de la pose
_mocap_ult_crudo = [None]


def _registrar_cambio_mocap(x, y, z, ahora):
    """Anota si la pose cruda cambió. Devuelve True si cambió tras estar
    congelada más de MOCAP_CONGELADO_S: Motive acaba de volver a verlo."""
    if (x, y, z) == _mocap_ult_crudo[0]:
        return False
    reenganche = (_mocap_ult_crudo[0] is not None
                  and ahora - _t_mocap_cambio[0] > MOCAP_CONGELADO_S)
    _mocap_ult_crudo[0] = (x, y, z)
    _t_mocap_cambio[0] = ahora
    return reenganche


def _filtrar_salto_mocap(x, y, z, reenganche=False):
    """
    Devuelve (x, y, z) a usar para `mocap_pose`. Si la muestra implica una
    velocidad mayor que MOCAP_SALTO_MAX_VEL respecto a la última aceptada,
    la descarta y repite la anterior: para este dron, ninguna cantidad de
    rechazos seguidos vuelve real un salto así, así que nunca se acepta por
    cansancio. Si los rechazos siguen más de MOCAP_SALTO_DURACION_PARO_S,
    activa el paro de emergencia: seguir mandando consignas contra una
    posición que no se puede fiar es peor que parar.
    `mocap_pose_cruda` no pasa por aquí: sigue siendo la referencia sin
    tocar para medir.
    `reenganche`: la pose venía congelada (Motive no veía el dron) y acaba
    de cambiar. Entonces la referencia vieja es la que está mal, no la
    muestra nueva: se acepta y se reancla, en vez de rechazar la posición
    buena para siempre (así acababa el vuelo de las 20:00 en paro).
    """
    g = _mocap_gate
    ahora = time.perf_counter()
    if reenganche and g['t'] is not None:
        print('  ↺ [MoCap] vuelve a rastrear el dron: se toma la posición nueva '
              '(la anterior estaba congelada).')
        g['x'], g['y'], g['z'], g['t'] = x, y, z, ahora
        g['t_primer_rechazo'] = None
        return x, y, z
    if g['t'] is None:
        g['x'], g['y'], g['z'], g['t'] = x, y, z, ahora
        g['t_primer_rechazo'] = None
        return x, y, z
    dt = ahora - g['t']
    salto = ((x - g['x']) ** 2 + (y - g['y']) ** 2 + (z - g['z']) ** 2) ** 0.5
    if dt > 0.0 and salto > MOCAP_SALTO_MAX_VEL * min(max(dt, MOCAP_SALTO_DT_MIN), MOCAP_SALTO_DT_MAX):
        if g['t_primer_rechazo'] is None:
            g['t_primer_rechazo'] = ahora
        duracion = ahora - g['t_primer_rechazo']
        print(f'  ⚠ [MoCap] muestra descartada: salto de {salto * 100:.1f} cm '
              f'en {dt * 1000:.1f} ms ({salto / max(dt, 1e-6):.1f} m/s), '
              f'{duracion:.2f} s rechazando')
        if duracion > MOCAP_SALTO_DURACION_PARO_S and not stop_event.is_set():
            print(f'  ✗ [MoCap] lleva {duracion:.1f} s dando saltos '
                  'imposibles: no es de fiar. Paro de emergencia.')
            activar_paro_emergencia('MoCap: saltos imposibles sostenidos')
        return g['x'], g['y'], g['z']
    g['x'], g['y'], g['z'], g['t'] = x, y, z, ahora
    g['t_primer_rechazo'] = None
    return x, y, z


scf_global      = None
last_ts         = None
stop_event      = threading.Event()


# ═══════════════════════════════════════════════════════════════
# GUARDAR / CARGAR RUTA
# ═══════════════════════════════════════════════════════════════

def guardar_ruta(nombre, puntos, ruta_suave):
    """Guarda el polígono y la ruta suave en un archivo JSON."""
    datos = {
        'nombre'     : nombre,
        'timestamp'  : datetime.now().isoformat(),
        'parametros' : {
            'PASO'    : PASO,
            'SUAVIDAD': SUAVIDAD,
            'N_SPLINE': N_SPLINE,
        },
        'poligono'   : puntos,
        'ruta_suave' : ruta_suave.tolist(),
    }
    ruta_archivo = os.path.join(RUTA_DIR, f'{nombre}.json')
    with open(ruta_archivo, 'w') as f:
        json.dump(datos, f, indent=2)
    print(f'✓ Ruta guardada en: {ruta_archivo}')
    return ruta_archivo

def listar_rutas():
    """Lista los archivos de ruta disponibles."""
    archivos = [f for f in os.listdir(RUTA_DIR) if f.endswith('.json')]
    return sorted(archivos)

def cargar_ruta(nombre_archivo):
    """Carga una ruta desde un archivo JSON."""
    ruta_archivo = os.path.join(RUTA_DIR, nombre_archivo)
    with open(ruta_archivo, 'r') as f:
        datos = json.load(f)
    puntos     = datos['poligono']
    ruta_suave = np.array(datos['ruta_suave'])
    params     = datos.get('parametros', {})
    print(f'✓ Ruta cargada: {datos["nombre"]} ({datos["timestamp"]})')
    print(f'  Parámetros: PASO={params.get("PASO")}, SUAVIDAD={params.get("SUAVIDAD")}, N_SPLINE={params.get("N_SPLINE")}')
    print(f'  Waypoints: {len(ruta_suave)}')
    return puntos, ruta_suave

def menu_inicio():
    """
    Muestra menú al inicio:
      1 → dibujar nueva ruta
      2 → cargar ruta existente
    Retorna (puntos_poligono, ruta_suave)
    """
    print('\n' + '═' * 50)
    print('  CRAZYFLIE — COBERTURA DE ÁREA')
    print('═' * 50)
    print('  1. Dibujar nueva área y generar ruta')
    print('  2. Cargar ruta guardada')
    print('─' * 50)

    archivos = listar_rutas()
    if not archivos:
        print('  (No hay rutas guardadas aún)')
    else:
        print(f'  Rutas disponibles ({len(archivos)}):')
        for i, a in enumerate(archivos):
            print(f'    [{i+1}] {a}')
    print('─' * 50)

    while True:
        opcion = input('Elige opción (1/2): ').strip()
        if opcion == '1':
            return None, None  # señal para dibujar nueva
        elif opcion == '2':
            if not archivos:
                print('No hay rutas guardadas. Dibuja una primero.')
                return None, None
            idx = input(f'Número de ruta (1-{len(archivos)}): ').strip()
            try:
                idx = int(idx) - 1
                if 0 <= idx < len(archivos):
                    return cargar_ruta(archivos[idx])
                else:
                    print('Número inválido.')
            except ValueError:
                print('Ingresa un número.')
        else:
            print('Opción inválida.')


# ═══════════════════════════════════════════════════════════════
# BLOQUE 1 — DIBUJO DEL POLÍGONO
# ═══════════════════════════════════════════════════════════════

def onclick(event):
    if event.xdata and event.ydata:
        puntos_clicks.append((event.xdata, event.ydata))
        ax_draw.scatter(event.xdata, event.ydata, color='red', zorder=5, s=50)
        if len(puntos_clicks) > 1:
            ax_draw.plot(
                [puntos_clicks[-2][0], puntos_clicks[-1][0]],
                [puntos_clicks[-2][1], puntos_clicks[-1][1]],
                'b-', linewidth=1.5
            )
        fig_draw.canvas.draw()

def onkey(event):
    if event.key == 'enter':
        plt.close(fig_draw)
        return
    # ── Imagen de fondo, sin cerrar la ventana ──
    # 'f' abre el explorador y cambia la foto en el sitio; 'g' la quita.
    # Así se pueden ir probando cultivos distintos sin renombrar archivos
    # ni volver a arrancar el programa. Los puntos que ya llevas marcados
    # NO se tocan: la imagen es una capa aparte (zorder=0).
    if event.key in ('f', 'F'):
        ruta = _elegir_fondo_dialogo()
        if ruta:
            if _poner_fondo(ax_draw, ruta):
                _titulo_dibujo(ax_draw, os.path.basename(ruta))
                fig_draw.canvas.draw_idle()
        return
    if event.key in ('g', 'G'):
        _quitar_fondo(olvidar=True)
        _titulo_dibujo(ax_draw, None)
        fig_draw.canvas.draw_idle()
        print('[fondo] Imagen quitada.')
        return
    # 'r' gira la imagen 90°. Si la foto viene en vertical y tu área es
    # horizontal (o al revés), esto la encaja sin deformarla: al girar se
    # intercambian también el ancho y el alto del rectángulo.
    if event.key in ('r', 'R'):
        global _fondo_rot
        if not _fondo_ruta_actual:
            print("[fondo] No hay imagen que girar (pulsa 'f' primero).")
            return
        _fondo_rot = (_fondo_rot + 90) % 360
        if _poner_fondo(ax_draw, _fondo_ruta_actual):
            _titulo_dibujo(ax_draw, os.path.basename(_fondo_ruta_actual))
            fig_draw.canvas.draw_idle()
            print(f'[fondo] Girada a {_fondo_rot}°.')
        return

def _extent_fondo(intercambiar=False):
    """
    Rectángulo (x_min, x_max, y_min, y_max) en metros de la imagen.

    Con `intercambiar` se cambian ancho y alto: es lo que hace falta cuando
    la imagen está girada 90°, para que no se deforme.
    """
    if FONDO_EXTENT is not None:
        x0, x1, y0, y1 = (float(v) for v in FONDO_EXTENT)
        if intercambiar:
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            a, h = (x1 - x0), (y1 - y0)
            return (cx - h / 2.0, cx + h / 2.0, cy - a / 2.0, cy + a / 2.0)
        return (x0, x1, y0, y1)
    cx, cy = FONDO_CENTRO
    ancho, alto = FONDO_ANCHO_M, FONDO_ALTO_M
    if intercambiar:
        ancho, alto = alto, ancho
    return (cx - ancho / 2.0, cx + ancho / 2.0,
            cy - alto / 2.0,  cy + alto / 2.0)


def _ruta_fondo():
    """
    De dónde sale la imagen: de `--fondo <archivo>` en la línea de órdenes
    o, si no está, de la constante FONDO_IMAGEN.

    Se lee sys.argv aquí y no en el parser de nadie a propósito: así esta
    opción no toca el arranque de ningún archivo.
    """
    try:
        if '--fondo' in sys.argv:
            i = sys.argv.index('--fondo')
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('-'):
                return sys.argv[i + 1]
            # `--fondo` sin nombre de archivo = "pregúntame cuál".
            return '__preguntar__'
    except Exception:
        pass
    return FONDO_IMAGEN


# Artistas de la imagen de fondo, para poder quitarla y poner otra sin
# tocar ni los clics ni los ejes.
_fondo_artistas = []
_fondo_rot = 0             # grados de giro aplicados (0/90/180/270)
_fondo_ruta_actual = None  # qué imagen está puesta, para poder regirarla


def _quitar_fondo(olvidar=False):
    """Borra la imagen de fondo que hubiera, dejando todo lo demás."""
    global _fondo_artistas, _fondo_ruta_actual
    for art in _fondo_artistas:
        try:
            art.remove()
        except Exception:
            pass
    _fondo_artistas = []
    if olvidar:
        _fondo_ruta_actual = None


def _elegir_fondo_dialogo():
    """
    Abre el explorador de Windows para escoger la imagen.

    Es lo que evita tener que renombrar archivos: se elige el que sea, esté
    donde esté. Si el diálogo no se puede abrir —pasa si el backend de
    matplotlib no es el de Tk—, se dice y no pasa nada más.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        raiz = tk.Tk()
        raiz.withdraw()
        raiz.attributes('-topmost', True)
        ruta = filedialog.askopenfilename(
            title='Imagen del cultivo (vista cenital)',
            filetypes=[('Imágenes', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff'),
                       ('PNG', '*.png'), ('Todos', '*.*')])
        raiz.destroy()
        return ruta or None
    except Exception as e:
        print(f'[fondo] No se pudo abrir el explorador: {e}')
        print('        Usa  --fondo "ruta\\a\\la\\imagen.png"  al arrancar.')
        return None


def _poner_fondo(ax, ruta=None):
    """
    Dibuja la imagen del cultivo por debajo de todo. Si algo falla, lo dice
    y sigue: la pantalla de dibujo tiene que funcionar igual sin imagen.
    Devuelve True si la puso.

    `ruta` explícita manda; si no se da, se mira --fondo y FONDO_IMAGEN.
    """
    if ruta is None:
        ruta = _ruta_fondo()
    if not ruta:
        return False
    if not os.path.exists(ruta):
        print(f'[fondo] No encuentro la imagen: {ruta}')
        print('        Se dibuja sin fondo. (Ruta relativa a donde ejecutas.)')
        return False
    # ══════════════════════════════════════════════════════════════
    # LEER LA IMAGEN RESPETANDO LA ORIENTACIÓN EXIF
    # ══════════════════════════════════════════════════════════════
    # Las fotos de dron y de móvil se guardan casi siempre en horizontal, y
    # aparte llevan una etiqueta EXIF que dice "gírame 90°". El Explorador
    # de Windows y cualquier visor la respetan, así que TÚ la ves vertical.
    #
    # `matplotlib.image.imread` NO la respeta: devuelve los píxeles tal
    # cual. Resultado: la misma foto que ves vertical entraba aquí
    # horizontal, girada un cuarto de vuelta respecto a todo lo demás.
    #
    # SI.jpg es exactamente ese caso: 2816×1536 en disco, orientación EXIF
    # 8, o sea 1536×2816 al verla. Por eso la parte vertical salía
    # horizontal: no era cuestión de gusto, estaba girada de verdad.
    #
    # Se lee con Pillow y `exif_transpose`, que aplica la etiqueta. Si no
    # hay Pillow se cae a matplotlib y se avisa: ahí queda la tecla 'r'
    # para enderezarla a mano.
    img = None
    try:
        from PIL import Image, ImageOps
        with Image.open(ruta) as im:
            im = ImageOps.exif_transpose(im)      # ← aquí está el arreglo
            if im.mode not in ('RGB', 'RGBA', 'L'):
                im = im.convert('RGB')
            img = np.asarray(im)
    except ImportError:
        print('[fondo] Sin Pillow: no puedo leer la orientación EXIF.')
        print("        Si la foto sale girada, pulsa 'r'.  (pip install pillow)")
    except Exception as e:
        print(f'[fondo] Pillow no pudo con la imagen ({e}); pruebo matplotlib.')

    if img is None:
        try:
            import matplotlib.image as mpimg
            img = mpimg.imread(ruta)
        except Exception as e:
            print(f'[fondo] No se pudo leer la imagen: {e}')
            if str(ruta).lower().endswith(('.jpg', '.jpeg')):
                print('        Para JPG hace falta Pillow:  pip install pillow')
                print('        (o guarda la imagen como PNG).')
            return False

    _quitar_fondo()          # fuera la anterior, si había

    # ── ROTACIÓN ──
    # Una foto cenital puede venir en vertical y el área del laboratorio
    # ser horizontal (o al revés). Al girar 90° se INTERCAMBIAN también el
    # ancho y el alto del rectángulo: si no, la foto se metería a la fuerza
    # en un hueco con la proporción cambiada y saldría deformada.
    global _fondo_rot, _fondo_ruta_actual
    _fondo_ruta_actual = ruta
    giros = (int(_fondo_rot) // 90) % 4
    if giros:
        img = np.rot90(img, k=giros)

    x0, x1, y0, y1 = _extent_fondo(intercambiar=(giros % 2 == 1))
    # origin='upper': la primera fila de la imagen va arriba, que es como se
    # ve una foto cenital. Sin esto saldría del revés en Y.
    #
    # SIN `aspect=`, y esto importa: `imshow(aspect='auto')` no afecta solo
    # a la imagen, le cambia el aspecto AL EJE ENTERO. Con eso, un metro en
    # X dejaba de medir lo mismo que un metro en Y y la trayectoria salía
    # aplastada en cuanto se ponía una imagen. Sin el argumento, el eje
    # conserva su 'equal' y la imagen se estira sola al `extent` pedido.
    _fondo_artistas.append(
        ax.imshow(img, extent=(x0, x1, y0, y1), origin='upper',
                  alpha=float(FONDO_ALPHA), zorder=0,
                  interpolation='bilinear'))
    ax.set_aspect('equal')       # cinturón y tirantes
    if FONDO_BORDE:
        _fondo_artistas.extend(
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                    color='0.35', linewidth=1.0, linestyle='--', zorder=1))

    # La imagen se ESTIRA al rectángulo pedido, que es lo que hace también
    # un proyector. Si las proporciones no cuadran, el cultivo sale
    # deformado respecto a la foto, así que se avisa con el número.
    try:
        alto_px, ancho_px = img.shape[0], img.shape[1]
        asp_img = ancho_px / max(alto_px, 1)
        asp_rec = (x1 - x0) / max(y1 - y0, 1e-9)
        print(f'[fondo] {os.path.basename(ruta)}  {ancho_px}×{alto_px} px  '
              f'→  {x1-x0:.2f} × {y1-y0:.2f} m')
        if abs(asp_img - asp_rec) / max(asp_rec, 1e-9) > 0.05:
            print(f'        ⚠ La imagen es {asp_img:.2f}:1 y el rectángulo '
                  f'{asp_rec:.2f}:1, así que se estira.')
            print('        Si quieres que no se deforme, ajusta '
                  'FONDO_ANCHO_M / FONDO_ALTO_M')
            print(f'        a esa proporción (p. ej. {FONDO_ALTO_M*asp_img:.2f} '
                  f'× {FONDO_ALTO_M:.2f} m) o recorta la foto.')
    except Exception:
        pass
    return True


def _titulo_dibujo(ax, nombre_fondo):
    """Título de la pantalla de dibujo, con o sin imagen puesta."""
    if nombre_fondo:
        giro = f' ({_fondo_rot}°)' if _fondo_rot else ''
        pie = (f'fondo: {nombre_fondo}{giro}   ·   f = cambiar   ·   '
               'r = girar 90°   ·   g = quitar')
    else:
        pie = 'f = poner una imagen del cultivo de fondo'
    ax.set_title(
        'Clic para definir el área de cobertura  →  ENTER para continuar\n'
        f'Área de trabajo: {AREA_X} m × {AREA_Y} m\n'
        f'({pie})',
        fontsize=11)


def dibujar_poligono():
    global fig_draw, ax_draw
    fig_draw, ax_draw = plt.subplots(figsize=(8, 10))
    ax_draw.set_xlim(-2, AREA_X)
    ax_draw.set_ylim(-2.5, AREA_Y)
    # Fondo opcional. Va ANTES del título para que, si falla, el aviso salga
    # antes de que se abra la ventana. No cambia ejes, límites ni escala:
    # solo pinta por debajo (zorder=0).
    global _fondo_rot
    _quitar_fondo(olvidar=True)
    _fondo_rot = int(FONDO_ROTAR) % 360
    ruta_ini = _ruta_fondo()
    # `--fondo` a secas, sin nombre de archivo: abre el explorador.
    if ruta_ini == '__preguntar__':
        ruta_ini = _elegir_fondo_dialogo()
    con_fondo = _poner_fondo(ax_draw, ruta_ini) if ruta_ini else False
    _titulo_dibujo(ax_draw, os.path.basename(ruta_ini) if con_fondo else None)
    ax_draw.set_aspect('equal')
    ax_draw.grid(True, alpha=0.3)
    ax_draw.set_xlabel('X (m)')
    ax_draw.set_ylabel('Y (m)')
    fig_draw.canvas.mpl_connect('button_press_event', onclick)
    fig_draw.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

    if len(puntos_clicks) < 3:
        print('Menos de 3 puntos — usando polígono de ejemplo.')
        puntos_clicks[:] = [
            (0.6, 0.6), (3.4, 0.4), (3.8, 2.5),
            (2.8, 4.6), (1.2, 4.4), (0.4, 2.5)
        ]


# ═══════════════════════════════════════════════════════════════
# BLOQUE 2 — GENERACIÓN DE RUTA
# ═══════════════════════════════════════════════════════════════

def densificar_contorno(poligono, paso_denso=0.01):
    coords = list(poligono.exterior.coords)
    denso = []
    for i in range(len(coords) - 1):
        a = np.array(coords[i])
        b = np.array(coords[i + 1])
        dist = np.linalg.norm(b - a)
        n = max(2, int(np.ceil(dist / paso_denso)))
        segmento = [tuple(a + t * (b - a)) for t in np.linspace(0, 1, n, endpoint=False)]
        denso.extend(segmento)
    denso.append(coords[-1])
    return denso

def camino_contorno(p1, p2, contorno_denso):
    arr = np.array(contorno_denso)

    def closest_index(p):
        return int(np.argmin(np.sum((arr - np.array(p))**2, axis=1)))

    i1 = closest_index(p1)
    i2 = closest_index(p2)

    if i1 <= i2:
        camino_cw = contorno_denso[i1:i2+1]
    else:
        camino_cw = contorno_denso[i1:] + contorno_denso[:i2+1]

    if i1 >= i2:
        camino_ccw = contorno_denso[i2:i1+1][::-1]
    else:
        camino_ccw = (contorno_denso[i2:] + contorno_denso[:i1+1])[::-1]

    def longitud(pts):
        a = np.array(pts)
        return np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1))

    return camino_cw if longitud(camino_cw) <= longitud(camino_ccw) else camino_ccw

def generar_ruta(poligono, paso):
    ruta = []
    minx, miny, maxx, maxy = poligono.bounds
    paso_denso = paso * 0.1
    contorno_denso = densificar_contorno(poligono, paso_denso=paso_denso)

    ruta.extend(contorno_denso)
    ultimo = contorno_denso[0]
    y, direccion = miny, 1

    while y <= maxy:
        linea = LineString([(minx - 0.01, y), (maxx + 0.01, y)])
        inter = poligono.intersection(linea)
        if inter.is_empty:
            y += paso
            continue
        if inter.geom_type == 'LineString':
            segmentos = [inter]
        elif inter.geom_type == 'MultiLineString':
            segmentos = list(inter.geoms)
        else:
            y += paso
            continue

        segmentos.sort(key=lambda s: s.bounds[0])
        if direccion == -1:
            segmentos.reverse()

        for seg in segmentos:
            x1, _ = seg.coords[0]
            x2, _ = seg.coords[-1]
            n_pts = max(3, int(abs(x2 - x1) / paso_denso))
            xs = np.linspace(x1, x2, n_pts) if direccion == 1 else np.linspace(x2, x1, n_pts)
            pl = [(x, y) for x in xs]
            if np.linalg.norm(np.array(ultimo) - np.array(pl[0])) > paso * 0.5:
                ruta.extend(camino_contorno(ultimo, pl[0], contorno_denso))
            ruta.extend(pl)
            ultimo = pl[-1]

        direccion *= -1
        y += paso

    ruta.extend(camino_contorno(ultimo, contorno_denso[0], contorno_denso))
    return ruta

def suavizar_bspline(ruta, suavidad=SUAVIDAD, n_puntos=N_SPLINE):
    pts = np.array(ruta)
    mask = np.concatenate(([True], np.any(np.diff(pts, axis=0) != 0, axis=1)))
    pts = pts[mask]
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate(([True], dists > 1e-6))
    pts = pts[keep]
    s_factor = suavidad * len(pts)
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=s_factor, k=3)
    u = np.linspace(0, 1, n_puntos)
    xs, ys = splev(u, tck)
    return np.column_stack([xs, ys])


# ═══════════════════════════════════════════════════════════════
# BLOQUE 2B — PLANIFICACIÓN DEL YAW
# ═══════════════════════════════════════════════════════════════

def _wrap_pi(a):
    """Envuelve un ángulo al rango [-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def preparar_waypoints(ruta_2d, min_dist=MIN_DIST):
    """Filtra la ruta densa dejando waypoints separados al menos `min_dist`."""
    pts = np.asarray(ruta_2d, dtype=float)
    wps = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - wps[-1]) >= min_dist:
            wps.append(p)
    if np.linalg.norm(pts[-1] - wps[-1]) > 1e-6:
        wps.append(pts[-1])
    return np.array(wps)


def calcular_yaws(wps, lookahead=None, ventana=None, offset=None):
    """
    Yaw deseado por waypoint = tangente de la ruta.

    - `lookahead` waypoints hacia adelante: compensa el retardo que introduce
      SOLAPAMIENTO (el dron va siempre un poco "detrás" de su consigna).
    - np.unwrap hace la secuencia CONTINUA: sin esto, al pasar de +179° a
      -179° el dron giraría 358° por el lado largo.
    - Media móvil sobre el ángulo ya desenrollado para quitar picos.

    Devuelve ángulos ACUMULADOS (pueden pasar de 360°). Es intencional:
    cada incremento es < 180°, y el firmware envuelve el error de yaw.
    """
    lookahead = YAW_LOOKAHEAD if lookahead is None else lookahead
    ventana   = YAW_VENTANA   if ventana   is None else ventana
    offset    = YAW_OFFSET    if offset    is None else offset

    wps = np.asarray(wps, dtype=float)
    n = len(wps)
    if n < 2:
        return np.zeros(max(n, 1))

    la = max(1, int(lookahead))
    yaws = np.zeros(n)
    for i in range(n):
        j = min(i + la, n - 1)
        d = wps[j] - wps[i]
        if np.linalg.norm(d) < 1e-9:                    # waypoints repetidos
            d = wps[i] - wps[i - 1] if i > 0 else np.array([1.0, 0.0])
        yaws[i] = np.arctan2(d[1], d[0])
    yaws[-1] = yaws[-2]                                 # el último no tiene tangente

    yaws = np.unwrap(yaws)                              # continuidad angular

    v = int(ventana) | 1                                # fuerza ventana impar
    if v > 1:
        k = np.ones(v) / v
        yaws = np.convolve(np.pad(yaws, (v // 2, v // 2), mode='edge'), k, mode='valid')

    return yaws + offset


def planificar_yaw(wps, yaws, vel=None, yaw_inicial=None):
    """
    Convierte (waypoints, yaws) en una lista de acciones ejecutables:

      {'tipo': 'rotar', 'yaw', 'dur', 'espera', 'idx'}        → gira parado
      {'tipo': 'volar', 'x','y','yaw', 'dur', 'espera', 'idx'} → avanza girando

    Dos tiempos distintos, y aquí está la clave:
      · 'espera' = tiempo REAL antes de mandar el siguiente comando.
                   Es lo que fija la velocidad sobre el suelo y la velocidad
                   de giro:  v = dist/espera ,  w = dyaw/espera.
      · 'dur'    = duración que se le pide al go_to, mayor que 'espera'
                   (factor 1/(1-SOLAPAMIENTO)). Como el dron nunca termina
                   el tramo antes de recibir el siguiente, no frena en cada
                   waypoint y el vuelo sale continuo.

    Antes 'dur' hacía las dos cosas a la vez, y con SOLAPAMIENTO=0.6 el dron
    volaba 2.5x más rápido de lo pedido.

    Además, en los tramos con más giro se baja la velocidad (VEL_FACTOR_GIRO)
    para darle tiempo a la nariz a alinearse.

    El waypoint 0 no genera acción: se llega a él en la fase de aproximación.
    """
    vel = FLY_VELOCITY if vel is None else vel
    acciones = []
    yaw_cmd = float(yaws[0]) if yaw_inicial is None else float(yaw_inicial)
    prev = np.asarray(wps[0], dtype=float)

    for i, p in enumerate(np.asarray(wps, dtype=float)):
        if i == 0:
            prev = p
            continue

        dist = max(float(np.linalg.norm(p - prev)), 0.01)
        dyaw = _wrap_pi(float(yaws[i]) - yaw_cmd) if YAW_ACTIVO else 0.0

        # Giro grande → girar parado antes de avanzar
        if abs(dyaw) > YAW_UMBRAL_ROT:
            yaw_cmd += dyaw
            t_rot = max(abs(dyaw) / YAW_RATE_MAX, YAW_T_MIN_ROT)
            acciones.append({'tipo': 'rotar', 'yaw': yaw_cmd, 'dur': t_rot,
                             'espera': t_rot + YAW_PAUSA_ROT, 'idx': i})
            dyaw = 0.0
        else:
            yaw_cmd += dyaw

        # Velocidad reducida según lo cerrado del giro
        factor = 1.0 - (1.0 - VEL_FACTOR_GIRO) * min(abs(dyaw) / YAW_UMBRAL_ROT, 1.0)
        vel_local = max(vel * factor, 0.01)

        # La espera respeta a la vez la velocidad lineal y la angular
        espera = max(dist / vel_local, abs(dyaw) / YAW_RATE_MAX, 0.3)
        dur    = espera / max(1.0 - SOLAPAMIENTO, 0.05)

        acciones.append({'tipo': 'volar', 'x': float(p[0]), 'y': float(p[1]),
                         'yaw': yaw_cmd, 'dur': dur, 'espera': espera, 'idx': i})
        prev = p

    return acciones


def tiempo_estimado(acciones):
    return sum(a['espera'] for a in acciones)


def resumen_plan(wps, yaws, acciones):
    """
    Informa QUÉ está limitando el vuelo, para saber qué parámetro tocar.

    Cada tramo está limitado por una de dos cosas:
      · velocidad → manda FLY_VELOCITY (súbelo para ir más rápido)
      · yaw       → manda YAW_RATE_MAX (súbelo, o el tramo se alarga solo)

    Si ves "limitados por yaw: 0", tienes margen de sobra para subir la
    velocidad sin que la nariz se quede atrás.
    """
    prev = np.asarray(wps[0], dtype=float)
    ycmd = float(yaws[0])
    n_vel = n_yaw = 0
    w_max = 0.0
    dist_total = 0.0

    for a in acciones:
        if a['tipo'] == 'rotar':
            ycmd = a['yaw']
            continue
        p = np.array([a['x'], a['y']])
        d = float(np.linalg.norm(p - prev))
        dy = abs(a['yaw'] - ycmd)
        factor = 1.0 - (1.0 - VEL_FACTOR_GIRO) * min(dy / YAW_UMBRAL_ROT, 1.0)
        t_vel = d / max(FLY_VELOCITY * factor, 0.01)
        t_yaw = dy / YAW_RATE_MAX
        if t_yaw > t_vel:
            n_yaw += 1
        else:
            n_vel += 1
        w_max = max(w_max, dy / a['espera'])
        dist_total += d
        prev = p
        ycmd = a['yaw']

    t = tiempo_estimado(acciones)
    n_rot = sum(1 for a in acciones if a['tipo'] == 'rotar')
    print(f'  Recorrido: {dist_total:.2f} m en ~{t:.0f} s '
          f'({dist_total / max(t, 0.01):.3f} m/s medios)')
    print(f'  Tramos limitados por velocidad: {n_vel} | por yaw: {n_yaw} | '
          f'rotaciones en sitio: {n_rot}')
    print(f'  Giro máximo comandado: {np.degrees(w_max):.0f} °/s '
          f'(techo {np.degrees(YAW_RATE_MAX):.0f} °/s)')
    if n_yaw == 0 and np.degrees(w_max) < np.degrees(YAW_RATE_MAX) * 0.8:
        print('  → Hay margen de giro: puedes subir FLY_VELOCITY.')
    elif n_yaw > len(acciones) * 0.2:
        print('  → El yaw es el cuello de botella: sube YAW_RATE_MAX '
              'o baja MIN_DIST.')
    return t


def dibujar_preview_yaw(ax, ruta_2d, cada=1):
    """Dibuja flechas de rumbo y marca dónde el dron girará en el sitio.
    Devuelve la lista de artistas para poder borrarlos al recargar una ruta."""
    artistas = []
    if not YAW_ACTIVO:
        return artistas

    wps      = preparar_waypoints(ruta_2d, MIN_DIST)
    yaws     = calcular_yaws(wps)
    acciones = planificar_yaw(wps, yaws)

    L = MIN_DIST * 0.8
    q = ax.quiver(wps[::cada, 0], wps[::cada, 1],
                  np.cos(yaws[::cada]) * L, np.sin(yaws[::cada]) * L,
                  color='#ffaa00', width=0.003, alpha=0.85,
                  angles='xy', scale_units='xy', scale=1, zorder=7,
                  label='Rumbo comandado')
    artistas.append(q)

    rot_idx = [a['idx'] for a in acciones if a['tipo'] == 'rotar']
    if rot_idx:
        s = ax.scatter(wps[rot_idx, 0], wps[rot_idx, 1], s=90, facecolors='none',
                       edgecolors='#ff3355', linewidths=1.6, zorder=9,
                       label=f'Giro en sitio ({len(rot_idx)})')
        artistas.append(s)

    print(f'  Preview: {len(wps)} waypoints')
    resumen_plan(wps, yaws, acciones)
    return artistas


# ═══════════════════════════════════════════════════════════════
# BLOQUE 3 — SIMULACIÓN PURE PURSUIT
# ═══════════════════════════════════════════════════════════════

class DronPurePursuit:
    def __init__(self, waypoints):
        self.wp         = waypoints
        self.pos        = waypoints[0].copy().astype(float)
        self.vel        = np.zeros(2)
        self.idx        = 0
        self.completado = False
        self.prev_error = np.zeros(2)
        self.tray       = [self.pos.copy()]
        self.vels       = [0.0]
        self.headings   = [0.0]

    def _lookahead_point(self):
        n = len(self.wp)
        while self.idx < n - 1:
            if np.linalg.norm(self.wp[self.idx] - self.pos) < LOOKAHEAD:
                self.idx += 1
            else:
                break
        return self.wp[min(self.idx, n - 1)]

    def step(self):
        if self.completado:
            return
        fraccion = self.idx / max(len(self.wp) - 1, 1)
        if np.linalg.norm(self.pos - self.wp[-1]) < UMBRAL_FIN and fraccion > 0.90:
            self.completado = True
            return
        target = self._lookahead_point()
        error  = target - self.pos
        dist   = np.linalg.norm(error)
        vel_d  = np.clip(KP * dist, VEL_MIN, VEL_MAX)
        dir_   = error / dist if dist > 1e-6 else np.zeros(2)
        d_error = (error - self.prev_error) / DT
        self.prev_error = error.copy()
        vel_obj  = dir_ * vel_d + KD * d_error
        alpha    = 0.12
        self.vel = alpha * vel_obj + (1 - alpha) * self.vel
        self.pos = self.pos + self.vel * DT
        heading  = np.arctan2(self.vel[1], self.vel[0]) if np.linalg.norm(self.vel) > 1e-5 else self.headings[-1]
        self.tray.append(self.pos.copy())
        self.vels.append(np.linalg.norm(self.vel))
        self.headings.append(heading)

def icono_dron(ax, pos, heading, escala=ESCALA_DRON):
    elementos = []
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    brazos = np.array([[1,0],[-1,0],[0,1],[0,-1]], dtype=float) * escala
    brazos_w = (R @ brazos.T).T + pos
    for b in brazos_w:
        l, = ax.plot([pos[0], b[0]], [pos[1], b[1]], '-', color='#dddddd', linewidth=2.2, zorder=12)
        elementos.append(l)
    theta = np.linspace(0, 2*np.pi, 18)
    r = escala * 0.55
    for b in brazos_w:
        rx = b[0] + r * np.cos(theta)
        ry = b[1] + r * np.sin(theta)
        l, = ax.plot(rx, ry, '-', color='#00ccff', linewidth=1.5, zorder=12)
        elementos.append(l)
    sc = ax.scatter(*pos, color='#ff3333', s=80, zorder=13, edgecolors='white', linewidths=0.8)
    elementos.append(sc)
    return elementos

def _pedir_nombre_figura(fig_padre, tx_estado):
    """
    Usa tkinter para pedir el nombre — funciona aunque matplotlib esté activo.
    """
    import tkinter as tk
    from tkinter import simpledialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    nombre = simpledialog.askstring(
        "Guardar ruta",
        "Nombre para la ruta (sin espacios):",
        initialvalue="mi_ruta",
        parent=root
    )
    root.destroy()
    if nombre:
        return nombre.strip().replace(" ", "_")
    return None


def _selector_rutas(archivos, parent=None):
    """
    Usa tkinter para mostrar un listbox con las rutas guardadas.
    Si se pasa parent, usa Toplevel (comparte mainloop).
    Retorna el índice seleccionado o None si canceló.
    """
    import tkinter as tk

    seleccion = [None]

    if parent is not None:
        # Ventana hija — comparte el mainloop del padre
        win = tk.Toplevel(parent)
    else:
        win = tk.Tk()

    win.title('Cargar ruta')
    win.configure(bg='#111122')
    win.attributes('-topmost', True)
    win.resizable(False, False)

    tk.Label(win, text='Selecciona una ruta guardada:',
             bg='#111122', fg='white', font=('Arial', 10)).pack(pady=(12, 4), padx=16)

    frame = tk.Frame(win, bg='#111122')
    frame.pack(padx=16, pady=4)

    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
    listbox = tk.Listbox(
        frame, yscrollcommand=scrollbar.set,
        bg='#1a2233', fg='#88ccff', selectbackground='#224466',
        font=('Courier', 10), width=40, height=min(len(archivos), 10),
        activestyle='underline', borderwidth=0
    )
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH)

    for a in archivos:
        listbox.insert(tk.END, a.replace('.json', ''))

    listbox.selection_set(0)

    def _ok():
        idx = listbox.curselection()
        if idx:
            seleccion[0] = idx[0]
        win.destroy()

    def _cancelar():
        win.destroy()

    listbox.bind('<Double-Button-1>', lambda e: _ok())

    btn_frame = tk.Frame(win, bg='#111122')
    btn_frame.pack(pady=10)

    tk.Button(btn_frame, text='✓ Cargar', command=_ok,
              bg='#003311', fg='#00ff88', font=('Arial', 10, 'bold'),
              relief='flat', padx=12).pack(side=tk.LEFT, padx=6)

    tk.Button(btn_frame, text='✗ Cancelar', command=_cancelar,
              bg='#220000', fg='#ff4444', font=('Arial', 10),
              relief='flat', padx=12).pack(side=tk.LEFT, padx=6)

    if parent is not None:
        parent.wait_window(win)  # espera sin bloquear el mainloop padre
    else:
        win.mainloop()

    return seleccion[0]


def _menu_inicio_tk():
    """
    Ventana tkinter de inicio: Nueva ruta o Cargar ruta guardada.
    Retorna ('nueva', None, None) o ('cargar', puntos, ruta_suave)
    """
    import tkinter as tk
    from tkinter import messagebox
    resultado = [None]

    root = tk.Tk()
    root.title('Crazyflie — Cobertura de área')
    root.configure(bg='#0d0d1a')
    root.attributes('-topmost', True)
    root.resizable(False, False)

    tk.Label(root, text='CRAZYFLIE — COBERTURA DE ÁREA',
             bg='#0d0d1a', fg='#00aaff', font=('Arial', 13, 'bold')).pack(pady=(18, 4), padx=30)
    tk.Label(root, text='¿Qué deseas hacer?',
             bg='#0d0d1a', fg='#888899', font=('Arial', 10)).pack(pady=(0, 12))

    archivos = listar_rutas()

    def _nueva():
        resultado[0] = ('nueva', None, None)
        root.destroy()

    def _cargar():
        if not archivos:
            messagebox.showwarning('Sin rutas', 'No hay rutas guardadas aún.')
            return
        # Ocultar el menú principal y abrir el selector como hijo
        root.withdraw()
        idx = _selector_rutas(archivos, parent=root)
        if idx is not None:
            puntos, ruta = cargar_ruta(archivos[idx])
            resultado[0] = ('cargar', puntos, ruta)
            root.destroy()
        else:
            # Si canceló el selector, volver a mostrar el menú
            root.deiconify()

    btn_frame = tk.Frame(root, bg='#0d0d1a')
    btn_frame.pack(pady=8, padx=30)

    tk.Button(btn_frame, text='✏  Dibujar nueva área',
              command=_nueva, bg='#003311', fg='#00ff88',
              font=('Arial', 11, 'bold'), relief='flat',
              padx=18, pady=10, width=28).pack(pady=6)

    estado_cargar = 'normal' if archivos else 'disabled'
    lbl = f'Cargar ruta guardada ({len(archivos)} disponibles)' if archivos else 'Cargar ruta (ninguna guardada)'
    tk.Button(btn_frame, text=lbl,
              command=_cargar, bg='#221100', fg='#ffaa44',
              font=('Arial', 11), relief='flat', state=estado_cargar,
              padx=18, pady=10, width=28).pack(pady=6)

    tk.Button(btn_frame, text='Salir',
              command=root.destroy, bg='#1a0000', fg='#ff4444',
              font=('Arial', 10), relief='flat',
              padx=18, pady=6, width=28).pack(pady=(12, 6))

    root.mainloop()
    return resultado[0] if resultado[0] else ('salir', None, None)


def simular_y_mostrar(ruta_suave_in, poly_pts):
    """
    Retorna: ('volar', ruta) | ('nueva', None) | ('cancelar', None)
    """
    # Estado mutable para poder actualizar desde callbacks
    estado = {
        'ruta'     : ruta_suave_in.copy(),
        'accion'   : None,   # 'volar' | 'nueva' | 'cancelar'
        'poly_pts' : poly_pts,
    }

    def _correr_sim(ruta_suave):
        dron = DronPurePursuit(ruta_suave)
        while not dron.completado:
            dron.step()
        return dron

    print('Simulando Pure Pursuit...')
    dron  = _correr_sim(estado['ruta'])
    tray  = np.array(dron.tray)
    vels  = np.array(dron.vels)
    heads = np.array(dron.headings)
    n     = len(tray)
    poly  = np.array(poly_pts)

    STRIDE  = max(1, n // 600)
    indices = list(range(0, n, STRIDE))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    fig = plt.figure(figsize=(15, 8), facecolor='#0d0d1a')
    fig.suptitle('SIMULACIÓN 2D — Pure Pursuit + PD  |  Rumbo tangente',
                 color='white', fontsize=13, fontweight='bold', y=0.97)

    ax = fig.add_axes([0.03, 0.12, 0.68, 0.81])
    ax.set_facecolor('#0d0d1a')
    ax.set_aspect('equal')
    ax.grid(True, color='#1a1a2e', linewidth=0.7)
    ax.tick_params(colors='#555566')
    ax.set_xlabel('X (m)', color='#888899')
    ax.set_ylabel('Y (m)', color='#888899')

    ruta = estado['ruta']
    ax.fill(poly[:, 0], poly[:, 1], color='#0a2a1a', alpha=0.7)
    poly_line, = ax.plot(poly[:, 0], poly[:, 1], '-', color='#00ff88', linewidth=1.8, label='Área', zorder=2)
    ruta_line, = ax.plot(ruta[:, 0], ruta[:, 1], '--', color='#334455',
                         linewidth=0.9, alpha=0.7, label='Ruta planificada', zorder=3)
    sc_ini = ax.scatter(*ruta[0],  color='#00ff88', s=100, zorder=10, label='Inicio')
    sc_fin = ax.scatter(*ruta[-1], color='#ff6600', s=100, zorder=10, label='Fin')

    # ── Previsualización del rumbo comandado ──
    preview = dibujar_preview_yaw(ax, estado['ruta'])

    circulo_la = plt.Circle(tray[0], LOOKAHEAD, color='#ffff00',
                             fill=False, linewidth=0.8, alpha=0.35, zorder=4)
    ax.add_patch(circulo_la)
    pt_la,    = ax.plot([], [], 'o', color='#ffff00', markersize=6, alpha=0.8, zorder=8, label='Lookahead')
    tray_line, = ax.plot([], [], '-', color='#00aaff', linewidth=1.6, alpha=0.9, zorder=5, label='Trayectoria sim')
    dron_elems = []
    ax.legend(facecolor='#111122', edgecolor='#334455', labelcolor='white', fontsize=8, loc='upper right')

    pad = LOOKAHEAD * 3
    ax.set_xlim(poly[:, 0].min() - pad, poly[:, 0].max() + pad)
    ax.set_ylim(poly[:, 1].min() - pad, poly[:, 1].max() + pad)

    # Panel telemetría
    ax_t = fig.add_axes([0.73, 0.12, 0.25, 0.81])
    ax_t.set_facecolor('#111122')
    ax_t.axis('off')
    ax_t.set_title('TELEMETRÍA SIM', color='#00aaff', fontsize=11, fontweight='bold', pad=10)
    kw = dict(transform=ax_t.transAxes, va='top', fontsize=9)
    tx_pos  = ax_t.text(0.06, 0.93, '', color='white', **kw)
    tx_vel  = ax_t.text(0.06, 0.83, '', color='white', **kw)
    tx_head = ax_t.text(0.06, 0.73, '', color='white', **kw)
    tx_prog = ax_t.text(0.06, 0.63, '', color='white', **kw)
    ax_t.add_patch(plt.Rectangle((0.06, 0.54), 0.88, 0.035,
                   facecolor='#223344', edgecolor='#334455', transform=ax_t.transAxes))
    barra = ax_t.add_patch(plt.Rectangle((0.06, 0.54), 0.001, 0.035,
                            facecolor='#00aaff', transform=ax_t.transAxes))
    ax_t.text(0.06, 0.46, '── Parámetros físicos ──', color='#00aaff',
              fontsize=8, transform=ax_t.transAxes, va='top')
    for i, txt in enumerate([
        f'Altura vuelo : {HOVER_HEIGHT:.2f} m',
        f'Velocidad    : {FLY_VELOCITY:.2f} m/s',
        f'En curva     : {FLY_VELOCITY*VEL_FACTOR_GIRO:.2f} m/s',
        f'Paso zig-zag : {PASO:.2f} m',
        f'Waypoints    : {len(ruta)} pts',
        f'Yaw          : {"tangente" if YAW_ACTIVO else "fijo 0°"}',
        f'Giro máx     : {np.degrees(YAW_RATE_MAX):.0f} °/s',
    ]):
        ax_t.text(0.06, 0.39 - i * 0.055, txt, color='#556677',
                  fontsize=8, transform=ax_t.transAxes, va='top')

    # ── Botones: [💾 GUARDAR] [🔄 NUEVA RUTA] [✓ VOLAR] [✗] ──
    ax_g  = fig.add_axes([0.730, 0.02, 0.082, 0.06])
    ax_n  = fig.add_axes([0.818, 0.02, 0.090, 0.06])
    ax_v  = fig.add_axes([0.914, 0.02, 0.050, 0.06])
    ax_x  = fig.add_axes([0.968, 0.02, 0.028, 0.06])

    btn_g = plt.Button(ax_g, '[G] GUARDAR', color='#001133', hovercolor='#003399')
    btn_n = plt.Button(ax_n, '[N] NUEVA RUTA', color='#112200', hovercolor='#335500')
    btn_v = plt.Button(ax_v, '✓ VOLAR',   color='#003311', hovercolor='#00aa44')
    btn_x = plt.Button(ax_x, '✗',         color='#220000', hovercolor='#aa0000')

    btn_g.label.set_color('#4499ff'); btn_g.label.set_fontsize(8)
    btn_n.label.set_color('#aadd44'); btn_n.label.set_fontsize(8)
    btn_v.label.set_color('#00ff88'); btn_v.label.set_fontsize(8); btn_v.label.set_fontweight('bold')
    btn_x.label.set_color('#ff4444'); btn_x.label.set_fontsize(10)

    tx_estado = fig.text(0.73, 0.10, '', color='#aaaaaa', fontsize=8, va='bottom')

    ani_holder = [None]

    def _lanzar_animacion(tray_local, heads_local, vels_local, ruta_local):
        """Lanza (o relanza) la animación con los datos actuales."""
        nonlocal dron_elems
        n_local = len(tray_local)
        stride  = max(1, n_local // 600)
        idxs    = list(range(0, n_local, stride))
        if idxs[-1] != n_local - 1:
            idxs.append(n_local - 1)

        def update(fi):
            nonlocal dron_elems
            i = idxs[fi]
            nf = max(len(idxs) - 1, 1)
            tray_line.set_data(tray_local[:i+1, 0], tray_local[:i+1, 1])
            circulo_la.center = tray_local[i]
            dists  = np.linalg.norm(ruta_local - tray_local[i], axis=1)
            il     = int(np.argmin(dists))
            for j in range(il, len(ruta_local)):
                if np.linalg.norm(ruta_local[j] - tray_local[i]) >= LOOKAHEAD:
                    il = j; break
            pt_la.set_data([ruta_local[il, 0]], [ruta_local[il, 1]])
            for el in dron_elems:
                try: el.remove()
                except Exception: pass
            dron_elems = icono_dron(ax, tray_local[i], heads_local[i])
            prog = fi / nf * 100
            tx_pos.set_text(f'Pos: ({tray_local[i,0]:.3f}, {tray_local[i,1]:.3f}) m')
            tx_vel.set_text(f'Vel: {vels_local[i]:.4f} m/s')
            tx_head.set_text(f'Rumbo: {np.degrees(heads_local[i]):.1f}°')
            tx_prog.set_text(f'Progreso: {prog:.1f}%')
            barra.set_width(0.88 * prog / 100)
            return [tray_line, pt_la] + dron_elems

        if ani_holder[0]:
            ani_holder[0].event_source.stop()
        ani_holder[0] = animation.FuncAnimation(
            fig, update, frames=len(idxs),
            interval=25, blit=False, repeat=False
        )
        fig.canvas.draw_idle()

    # Lanzar animación inicial
    _lanzar_animacion(tray, heads, vels, estado['ruta'])

    def on_guardar(event):
        nombre = _pedir_nombre_figura(fig, tx_estado)
        if nombre:
            pts_lista = [[float(x), float(y)] for x, y in zip(poly_pts[:, 0], poly_pts[:, 1])]
            guardar_ruta(nombre, pts_lista, estado['ruta'])
            tx_estado.set_text(f'✓ Guardado: {nombre}.json')
            fig.canvas.draw_idle()

    def on_nueva(event):
        """Cargar una ruta guardada O volver al menú para dibujar."""
        nonlocal preview
        import tkinter as tk
        from tkinter import messagebox

        root2 = tk.Tk()
        root2.title('Nueva / Cargar')
        root2.configure(bg='#0d0d1a')
        root2.attributes('-topmost', True)
        root2.resizable(False, False)
        eleccion = [None]

        tk.Label(root2, text='¿Qué deseas hacer?',
                 bg='#0d0d1a', fg='white', font=('Arial', 10)).pack(pady=(14, 6), padx=20)

        def _dibujar():
            eleccion[0] = 'dibujar'
            root2.destroy()

        def _cargar_aqui():
            archivos = listar_rutas()
            if not archivos:
                messagebox.showwarning('Sin rutas', 'No hay rutas guardadas.')
                return
            idx = _selector_rutas(archivos)
            if idx is not None:
                puntos, nueva_ruta = cargar_ruta(archivos[idx])
                eleccion[0] = ('cargada', nueva_ruta)
            root2.destroy()

        def _cancelar2():
            root2.destroy()

        f = tk.Frame(root2, bg='#0d0d1a'); f.pack(padx=20, pady=4)
        tk.Button(f, text='✏  Dibujar nueva área', command=_dibujar,
                  bg='#003311', fg='#00ff88', font=('Arial', 10, 'bold'),
                  relief='flat', padx=12, pady=8, width=22).pack(pady=4)
        tk.Button(f, text='📂  Cargar ruta guardada', command=_cargar_aqui,
                  bg='#221100', fg='#ffaa44', font=('Arial', 10),
                  relief='flat', padx=12, pady=8, width=22).pack(pady=4)
        tk.Button(f, text='↩  Volver', command=_cancelar2,
                  bg='#111122', fg='#888899', font=('Arial', 9),
                  relief='flat', padx=12, pady=6, width=22).pack(pady=(8, 10))

        root2.mainloop()

        if eleccion[0] == 'dibujar':
            estado['accion'] = 'nueva'
            plt.close(fig)
        elif eleccion[0] is not None and eleccion[0][0] == 'cargada':
            nueva_ruta = eleccion[0][1]
            estado['ruta'] = nueva_ruta
            # Limpiar y redibujar
            tray_line.set_data([], [])
            ruta_line.set_data(nueva_ruta[:, 0], nueva_ruta[:, 1])
            sc_ini.set_offsets([nueva_ruta[0]])
            sc_fin.set_offsets([nueva_ruta[-1]])
            for art in preview:
                try: art.remove()
                except Exception: pass
            preview = dibujar_preview_yaw(ax, nueva_ruta)
            ax.set_xlim(nueva_ruta[:, 0].min() - pad, nueva_ruta[:, 0].max() + pad)
            ax.set_ylim(nueva_ruta[:, 1].min() - pad, nueva_ruta[:, 1].max() + pad)
            # Recalcular simulación con ruta nueva
            print('Recalculando simulación...')
            dron2 = _correr_sim(nueva_ruta)
            _lanzar_animacion(np.array(dron2.tray), np.array(dron2.headings),
                              np.array(dron2.vels), nueva_ruta)
            tx_estado.set_text('✓ Ruta cargada y simulación actualizada')
            fig.canvas.draw_idle()

    def on_volar(event):
        estado['accion'] = 'volar'
        plt.close(fig)

    def on_cancelar(event):
        estado['accion'] = 'cancelar'
        plt.close(fig)

    btn_g.on_clicked(on_guardar)
    btn_n.on_clicked(on_nueva)
    btn_v.on_clicked(on_volar)
    btn_x.on_clicked(on_cancelar)

    plt.show()
    return estado['accion'], estado['ruta']


# ═══════════════════════════════════════════════════════════════
# BLOQUE 4 — EJECUCIÓN FÍSICA
# ═══════════════════════════════════════════════════════════════

_mocap_n  = 0
_mocap_t0 = None
_t_ultimo_extpos = 0.0

# Última pose recibida y su número de secuencia. El hilo de envío solo manda
# cuando hay dato nuevo, así no repite la misma corrección una y otra vez.
_mocap_seq = 0
_mocap_ori = None

# Instantes de cada muestra de real_trajectory, y ventana en la que el dron
# estuvo REALMENTE recorriendo la ruta (sin despegue, aproximación ni
# aterrizaje). Sin esto, el error de seguimiento sale inflado: la
# aproximación al inicio son más de un metro de vuelo que no pertenece a la
# ruta, y al anclar por longitud de arco desplaza TODAS las muestras.
tiempos_trayectoria = []
VENTANA_RUTA = {'ini': None, 'fin': None}
_extpos_stats = {'enviados': 0, 'ms_total': 0.0, 'n': 0, 'saltados': 0}
_mocap_n_bruto = 0          # mensajes recibidos ANTES de filtrar
_ult_pos_aceptada = None    # para el filtro por contenido
# Instante del último mensaje del MoCap, para el perro guardián del lazo de
# Pure Pursuit: si la posición se queda vieja, hay que mandar velocidad cero
# en vez de seguir empujando a ciegas con el último comando.
_t_mocap_fresco = [0.0]

def _contar_hz_mocap(bruto=False):
    """
    Cadencia del MoCap. Se imprime cada 5 s.

    OJO con la diferencia, que costó cara: `rx` tiene que contarse ANTES
    del filtro de timestamp. Contándolo después, `rx` mostraba 1.0 Hz y
    parecía que el MoCap solo mandaba un mensaje por segundo, cuando en
    realidad llegaban ~90 Hz y el filtro tiraba el 99 %. El dato estaba
    ahí todo el tiempo; el panel lo escondía.

      bruto=True  → mensaje recibido (antes de filtrar)
      bruto=False → mensaje aceptado (pasó el filtro)
    """
    global _mocap_n, _mocap_t0, _mocap_n_bruto
    if not MOSTRAR_HZ_MOCAP:
        return
    if bruto:
        _mocap_n_bruto += 1
        return
    _mocap_n += 1
    ahora = time.time()
    if _mocap_t0 is None:
        _mocap_t0 = ahora
    elif ahora - _mocap_t0 >= 5.0:
        hz = _mocap_n / (ahora - _mocap_t0)
        aviso = ''
        st = _extpos_stats
        ventana = ahora - _mocap_t0
        extra = ''
        hz_tx = None
        if st['n']:
            hz_tx = st['enviados'] / ventana
            ms = st['ms_total'] / st['n'] if st['ms_total'] else 0.0
            extra = f' | al EKF {hz_tx:4.1f} Hz'
            if ms:
                extra += f', {ms:.1f} ms/envio'
            if hz_tx < MOCAP_HZ_MAX * 0.7 and MOCAP_HZ_MAX > 0:
                extra += f' (pedidos {MOCAP_HZ_MAX:.0f})'
        st.update(enviados=0, ms_total=0.0, n=0, saltados=0)
        if hz_tx is not None and hz_tx < 25:
            aviso = '  <- EKF POR DEBAJO DE 30 Hz'
        hz_bruto = _mocap_n_bruto / ventana
        # Si el filtro está tirando la mayoría, decirlo: el cuello de
        # botella es el filtro, no la red.
        if hz_bruto > hz * 2.0:
            desecho = 100.0 * (1.0 - hz / hz_bruto)
            print(f'[MoCap] llegan {hz_bruto:5.1f} Hz | aceptados {hz:4.1f} Hz'
                  f' ({desecho:.0f} % descartado por el filtro)'
                  f'{extra}{aviso}')
        else:
            print(f'[MoCap] rx {hz:.1f} Hz{extra}{aviso}')
        _mocap_n, _mocap_t0 = 0, ahora
        _mocap_n_bruto = 0


class HiloExtpos(threading.Thread):
    """
    Manda la pose al EKF a ritmo fijo, desde su PROPIO hilo.

    Por qué existe: antes el `send_extpos` se hacía dentro del callback de
    MQTT. Si la radio tardaba en aceptar el paquete, ese retraso frenaba la
    lectura del socket MQTT, y la frecuencia efectiva del MoCap se desplomaba
    por debajo de lo que el limitador pedía. O sea: la radio marcaba el ritmo
    al que se leía la red, que es justo al revés de como debe ser.

    Ahora el callback solo apunta la pose y vuelve; este hilo coge la ÚLTIMA
    disponible y la manda. Si la radio se atasca, se pierden correcciones
    intermedias (que es lo correcto: la pose vieja no sirve), pero el MoCap
    se sigue leyendo a toda velocidad.
    """

    def __init__(self, hz):
        super().__init__(daemon=True, name='extpos')
        self.periodo = 1.0 / hz if hz > 0 else 0.0
        self._ultimo_seq = -1

    def run(self):
        global USAR_EXTPOS_DIRECTO
        afinar_temporizador_windows(True)
        siguiente = time.time()
        while not stop_event.is_set():
            try:
                seq = _mocap_seq
                if scf_global is not None and seq != self._ultimo_seq:
                    self._ultimo_seq = seq
                    x, y, z = mocap_pose['x'], mocap_pose['y'], mocap_pose['z']
                    t_env = time.time()
                    if USAR_ORIENTACION_MOCAP and _mocap_ori is not None:
                        scf_global.cf.extpos.send_extpose(
                            x, y, z,
                            float(_mocap_ori['x']), float(_mocap_ori['y']),
                            float(_mocap_ori['z']), float(_mocap_ori['w']))
                    elif USAR_EXTPOS_DIRECTO:
                        try:
                            scf_global.cf.extpos.send_extpos(x, y, z)
                        except Exception:
                            USAR_EXTPOS_DIRECTO = False
                            set_position(scf_global, x, y, z)
                    else:
                        set_position(scf_global, x, y, z)
                    dt = (time.time() - t_env) * 1000.0
                    _extpos_stats['enviados'] += 1
                    _extpos_stats['ms_total'] += dt
                    _extpos_stats['n'] += 1
                else:
                    _extpos_stats['saltados'] += 1
            except Exception as e:
                if not stop_event.is_set():
                    print(f'[extpos] error: {e}')
                time.sleep(0.05)

            # Planificación por fecha límite: el periodo no se mide desde
            # el inicio de ESTA vuelta sino desde una rejilla fija, así el
            # tiempo de procesado no se va acumulando como retraso.
            siguiente += self.periodo
            resto = siguiente - time.time()
            if resto > 0:
                time.sleep(resto)
            elif resto < -0.5:
                siguiente = time.time()   # nos quedamos muy atrás: resincronizar


def on_message(client, userdata, msg):
    global scf_global, last_ts, _mocap_n, _mocap_t0, USAR_EXTPOS_DIRECTO
    global _mocap_ori, _mocap_seq_cruda
    if stop_event.is_set():
        return
    try:
        data   = json.loads(msg.payload.decode())
        pose   = data['payload']['pose']
        pos    = pose['position']
        ori    = pose.get('orientation', None)
        ts_str = data.get('ts', None)

        # Guardar la orientación SIEMPRE, aunque no se envíe al EKF: sirve
        # para medir cuánto se ha desviado el yaw estimado del real, que es
        # lo que decide en qué dirección empuja el controlador.
        #
        # ERROR MÍO, corregido: yo solo miraba `pose['orientation']` con
        # claves x/y/z/w, no lo encontraba nunca, y el informe repetía
        # vuelo tras vuelo "el mensaje del MoCap no trae orientación".
        # El puente la manda en `pose['rotation']` con claves qx/qy/qz/qw
        # (así la lee flight_purepursuit.py, y sus métricas imprimen un yaw
        # real: -168.7°). El yaw se podía medir desde el principio.
        # Ahora se aceptan las dos formas.
        if ori is None:
            rot = pose.get('rotation', None)
            if isinstance(rot, dict) and 'qw' in rot:
                ori = {'x': float(rot['qx']), 'y': float(rot['qy']),
                       'z': float(rot['qz']), 'w': float(rot['qw'])}
        if ori is not None:
            _mocap_ori = ori
            mocap_pose['yaw'] = float(np.arctan2(
                2.0 * (ori['w'] * ori['z'] + ori['x'] * ori['y']),
                1.0 - 2.0 * (ori['y'] * ori['y'] + ori['z'] * ori['z'])))

        # Registro para el análisis posterior: se hace ANTES del filtro, así
        # queda la trayectoria completa a la cadencia real del MoCap aunque
        # al EKF solo le lleguen unas pocas correcciones. No manda nada.
        if REGISTRAR_TRAYECTORIA_COMPLETA:
            real_trajectory.append([float(pos['x']), float(pos['y']),
                                    float(pos['z'])])
            tiempos_trayectoria.append(time.time())

        # Pose sin filtrar, para MEDIR (no se envía al dron). Va antes del
        # filtro a propósito: es la única referencia con la que se puede
        # saber de verdad dónde estaba el dron en cada instante.
        mocap_pose_cruda['x'] = float(pos['x'])
        mocap_pose_cruda['y'] = float(pos['y'])
        mocap_pose_cruda['z'] = float(pos['z'])
        # Filtrada UNA vez: lo que se manda al EKF y al lazo de control no
        # debe incluir un salto de un solo cuadro (ver el comentario largo
        # junto a MOCAP_SALTO_MAX_VEL). `mocap_pose_cruda` de arriba, en
        # cambio, sigue siendo la referencia sin tocar para medir.
        _reeng = _registrar_cambio_mocap(mocap_pose_cruda['x'],
                                         mocap_pose_cruda['y'],
                                         mocap_pose_cruda['z'],
                                         time.perf_counter())
        _mx, _my, _mz = _filtrar_salto_mocap(mocap_pose_cruda['x'],
                                             mocap_pose_cruda['y'],
                                             mocap_pose_cruda['z'],
                                             reenganche=_reeng)
        _mocap_seq_cruda += 1
        _t_mocap_fresco[0] = time.perf_counter()
        _contar_hz_mocap(bruto=True)     # cadencia REAL de llegada

        # ── SEPARAR LAS DOS CADENCIAS ──
        # Esta es la pieza que me faltaba, y flight_purepursuit.py la tiene:
        # son DOS consumidores con necesidades distintas.
        #   · El lazo de control del PC necesita la posición lo más fresca
        #     posible, a los ~93 Hz que llegan. Filtrarla lo ciega.
        #   · El EKF del dron necesita correcciones ESPACIADAS, porque cada
        #     una es un paquete de radio que compite con los setpoints.
        # Mi filtro 'contenido' estrangulaba las dos a la vez, que es
        # justamente lo que no hay que hacer. Aquí `mocap_pose` se actualiza
        # siempre y el limitador de abajo solo decide cuándo se MANDA.
        if MODO_EJECUCION == 'pursuit':
            mocap_pose['x'] = _mx
            mocap_pose['y'] = _my
            mocap_pose['z'] = _mz

        # ── MODO 'contenido': deduplicar por la POSICIÓN, no por la fecha ──
        # El motivo: `ts` viene con resolución de UN SEGUNDO, así que el
        # filtro estricto solo puede dejar pasar un mensaje por segundo
        # aunque lleguen noventa. Pero el propósito del filtro no es leer
        # la fecha: es no reenviar dos veces la misma muestra. Y eso se
        # sabe mirando la posición, que no depende de `ts` para nada.
        # Si la posición cambió, es una muestra nueva, diga lo que diga la
        # fecha. Si es idéntica al micrómetro, es la misma repetida.
        if MODO_FILTRO_TS == 'contenido':
            global _ult_pos_aceptada
            p = (float(pos['x']), float(pos['y']), float(pos['z']))
            if _ult_pos_aceptada is not None:
                d2 = ((p[0] - _ult_pos_aceptada[0]) ** 2 +
                      (p[1] - _ult_pos_aceptada[1]) ** 2 +
                      (p[2] - _ult_pos_aceptada[2]) ** 2)
                if d2 < MOCAP_MIN_MOV ** 2:
                    return           # misma muestra repetida
            _ult_pos_aceptada = p
        elif ts_str and MODO_FILTRO_TS != 'ninguno':
            msg_time = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            if last_ts is not None:
                if MODO_FILTRO_TS == 'estricto' and msg_time <= last_ts:
                    return          # comportamiento original
                if MODO_FILTRO_TS == 'tolerante' and msg_time < last_ts:
                    return          # solo rechaza los genuinamente viejos
            last_ts = msg_time

        global _t_ultimo_extpos, _mocap_seq

        # ── MODO 'hilo': el callback NO toca la radio ──
        # Solo deja la pose en una variable y vuelve. Así paho puede leer el
        # socket a toda velocidad y la frecuencia del MoCap deja de depender
        # de lo que tarde la radio en aceptar un paquete.
        if MOCAP_MODO_ENVIO == 'hilo':
            mocap_pose['x'] = _mx
            mocap_pose['y'] = _my
            mocap_pose['z'] = _mz
            _mocap_ori = ori
            _mocap_seq += 1
            if not REGISTRAR_TRAYECTORIA_COMPLETA:
                real_trajectory.append([mocap_pose['x'], mocap_pose['y'],
                                        mocap_pose['z']])
            _contar_hz_mocap()
            return

        # ── MODO 'callback' ──
        # La pose local se actualiza SIEMPRE; lo que se limita es cuántas
        # veces se toca la radio. perf_counter porque es monótono y de alta
        # resolución: time.time() en Windows salta de 15 en 15 ms.
        mocap_pose['x'] = _mx
        mocap_pose['y'] = _my
        mocap_pose['z'] = _mz
        if not REGISTRAR_TRAYECTORIA_COMPLETA:
            real_trajectory.append([mocap_pose['x'], mocap_pose['y'],
                                    mocap_pose['z']])
        _contar_hz_mocap()

        # Limitador por FECHA LÍMITE ACUMULATIVA, no por "tiempo desde el
        # último envío". La diferencia importa: paho entrega los mensajes a
        # ráfagas (varios casi a la vez y luego un hueco). Con el criterio
        # antiguo, de cada ráfaga pasaba UNO solo y el resto caía dentro de
        # la ventana, así que con 68 Hz de entrada y 50 pedidos salían 20.
        # Avanzando la fecha límite en pasos fijos, si venimos retrasados el
        # siguiente mensaje de la ráfaga también pasa, y la media sale bien.
        ahora_lim = time.perf_counter()
        if MOCAP_HZ_MAX > 0:
            if ahora_lim < _t_ultimo_extpos:
                return
            periodo = 1.0 / MOCAP_HZ_MAX
            _t_ultimo_extpos += periodo
            if _t_ultimo_extpos < ahora_lim - 0.5:
                _t_ultimo_extpos = ahora_lim + periodo   # resincronizar
        _extpos_stats['enviados'] += 1
        _extpos_stats['n'] += 1
        if scf_global and not stop_event.is_set():
            if USAR_ORIENTACION_MOCAP and ori is not None:
                # Pose completa: el EKF corrige también el yaw (no deriva)
                scf_global.cf.extpos.send_extpose(
                    mocap_pose['x'], mocap_pose['y'], mocap_pose['z'],
                    float(ori['x']), float(ori['y']),
                    float(ori['z']), float(ori['w'])
                )
            elif USAR_EXTPOS_DIRECTO:
                # Igual que set_position pero sin imprimir nada en cada mensaje
                try:
                    scf_global.cf.extpos.send_extpos(
                        mocap_pose['x'], mocap_pose['y'], mocap_pose['z'])
                except Exception:
                    USAR_EXTPOS_DIRECTO = False   # no soportado → usar el de siempre
                    set_position(scf_global, mocap_pose['x'],
                                 mocap_pose['y'], mocap_pose['z'])
            else:
                set_position(scf_global, mocap_pose['x'], mocap_pose['y'], mocap_pose['z'])

    except Exception as e:
        print(f'Error MQTT: {e}')

mqtt_estado = {'conectado': False, 'error': None, 'intentos': 0}


def _ip_local_hacia(destino):
    """
    IP local que el sistema usaría para hablar con `destino`.
    No envía nada: solo consulta la tabla de rutas con un socket UDP.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect((destino, 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def diagnostico_red():
    """
    Explica POR QUÉ no se llega al broker del MoCap.

    El caso que más pasa en este montaje: la laptop se asocia al punto de
    acceso propio del AI-deck (192.168.4.x) y con eso sale de la red del
    laboratorio, donde vive el broker. La cámara funciona y el MoCap no.
    Aquí se detecta comparando subredes, sin adivinar.
    """
    ip = _ip_local_hacia(MQTT_BROKER)
    print('─' * 62)
    print('  NO HAY MOCAP — DIAGNÓSTICO DE RED')
    print('─' * 62)
    print(f'  Broker MQTT esperado : {MQTT_BROKER}:{MQTT_PORT}')
    print(f'  IP local hacia él    : {ip or "(no hay ruta)"}')

    red_broker = '.'.join(MQTT_BROKER.split('.')[:3])
    red_local = '.'.join(ip.split('.')[:3]) if ip else None

    if red_local and red_local != red_broker:
        print(f'  Subred local {red_local}.x  ≠  subred del broker {red_broker}.x')
        print()
        if red_local == '.'.join(VISION_IP.split('.')[:3]):
            print('  → ESTÁS EN LA RED DEL AI-DECK, NO EN LA DEL LABORATORIO.')
            print('    Al conectarte al punto de acceso del deck saliste de la')
            print('    red donde está el MoCap. Por eso la cámara va y el')
            print('    MoCap no: son dos redes distintas y la laptop solo')
            print('    puede estar en una a la vez por WiFi.')
        else:
            print('  → La laptop está en otra red. Revisa a qué WiFi está')
            print('    conectada.')
    else:
        print('  Las subredes coinciden: el problema no es el enrutado.')
        print('  → Revisa que el broker esté encendido y escuchando en ese')
        print(f'    puerto, y que el tema "{MQTT_TOPIC}" se esté publicando.')

    print()
    print('  CÓMO TENER CÁMARA Y MOCAP A LA VEZ (elige una):')
    print('   1. AI-deck en modo ESTACIÓN, unido al WiFi del laboratorio.')
    print('      Es la buena: todo en una red. Pon VISION_IP a la IP que le')
    print('      dé el router y la laptop no se mueve de la red del MoCap.')
    print('   2. Dos interfaces: cable Ethernet a la red del laboratorio y')
    print('      WiFi al AP del deck. Funciona sin tocar el firmware.')
    print('   3. Sin cámara por hoy: VISION_ACTIVA = False y vuelas como')
    print('      siempre, con el MoCap en su red.')
    print('─' * 62)


def start_mqtt():
    """
    Cliente MQTT del MoCap, con reintentos.

    Antes, si el broker no respondía, este hilo moría con un traceback y ya
    no había MoCap en toda la sesión. Ahora explica qué pasa y sigue
    reintentando: si arreglas la red a mitad, se reengancha solo.
    """
    espera = 3.0
    while not stop_event.is_set():
        mqtt_estado['intentos'] += 1
        try:
            client = mqtt.Client()
            client.on_message = on_message
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            client.subscribe(MQTT_TOPIC)
            mqtt_estado['conectado'] = True
            mqtt_estado['error'] = None
            print(f'✓ MoCap conectado a {MQTT_BROKER}:{MQTT_PORT} '
                  f'(tema {MQTT_TOPIC})')
            client.loop_forever()
            mqtt_estado['conectado'] = False
            if not stop_event.is_set():
                print('⚠ Se perdió la conexión con el broker del MoCap.')
        except Exception as e:
            mqtt_estado['conectado'] = False
            mqtt_estado['error'] = str(e)
            if mqtt_estado['intentos'] == 1:
                print(f'\n✗ No se pudo conectar al broker del MoCap: {e}')
                diagnostico_red()
            else:
                print(f'  (reintento MoCap {mqtt_estado["intentos"]}: {e})')

        for _ in range(int(espera / 0.1)):
            if stop_event.is_set():
                return
            time.sleep(0.1)
        espera = min(espera * 1.5, 15.0)

# ─────────────────────────────────────────────
# PARO DE EMERGENCIA
# ─────────────────────────────────────────────
# Un ÚNICO punto de entrada, al que llaman TODOS los que pueden dispararlo:
#   · el hilo del teclado de la consola (msvcrt)
#   · la ventana de la cámara (cv2.waitKey)
# Es idempotente y no bloquea: si ya se activó, no hace nada.

_paro_lock = threading.Lock()

def activar_paro_emergencia(origen='consola'):
    """Corta motores y marca stop_event. Seguro de llamar desde cualquier hilo."""
    with _paro_lock:
        if stop_event.is_set():
            return
        print(f'\n🚨 PARO DE EMERGENCIA ACTIVADO  [{origen}]')
        stop_event.set()
    _cortar_motores()

# Sistema de visión en curso. Lo pone SistemaVision.iniciar(). Sirve para
# que el teclado de la consola pueda reabrir la ventana de la cámara: una vez
# cerrada, el teclado de OpenCV ya no existe, así que si la única forma de
# reabrirla estuviera en la propia ventana, no habría ninguna.
VISION_EN_CURSO = None

def emergency_listener():
    print("Presiona 'q' para paro de emergencia (consola o ventana de la camara)"
          "   ·   'v' = reabrir la ventana de la camara")
    while not stop_event.is_set():
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if key == 'q':
                activar_paro_emergencia('consola')
                break
            if key == 'v':
                v = VISION_EN_CURSO
                if v is None:
                    print('[cam] No hay visión en marcha.')
                elif v.ventana_abierta:
                    print('[cam] La ventana ya está abierta.')
                else:
                    v.mostrar_ventana()
                    print('[cam] Reabriendo la ventana de la cámara...')
        time.sleep(0.05)

def _cortar_motores():
    global scf_global
    if scf_global:
        led_off(scf_global)          # que no se quede encendido tras el paro
        # Abortar la trayectoria del firmware ANTES de cortar motores: si no,
        # el planificador seguiría vivo por dentro.
        try:
            scf_global.cf.high_level_commander.stop()
        except Exception:
            pass
        try:
            scf_global.cf.commander.send_stop_setpoint()
            print('✓ Motores apagados')
        except Exception as e:
            print(f'Error apagando motores: {e}')


# ─────────────────────────────────────────────
# CONTROL DEL COLOR LED DECK
# ─────────────────────────────────────────────
# Nunca lanzan excepción: si el deck no está montado, avisan una vez y el
# vuelo continúa igual. Un LED no debe poder tumbar una misión.

_led_estado = {'disponible': None, 'encendido': False}

def _led_hay_deck(scf):
    """¿Está montado el Color LED Deck? Se comprueba una sola vez."""
    if _led_estado['disponible'] is not None:
        return _led_estado['disponible']
    hay = False
    try:
        # Si el deck no está, el parámetro no existe en el TOC del firmware
        elem = scf.cf.param.toc.get_element_by_complete_name(f'{LED_GRUPO}.wrgb8888')
        hay = elem is not None
    except Exception:
        hay = False
    _led_estado['disponible'] = hay
    if not hay:
        print('⚠ Color LED Deck no detectado — se vuela sin LED.')
    return hay

def led_init(scf):
    """Fija el brillo y deja el LED apagado. Llamar tras conectar."""
    if not LED_ACTIVO or scf is None or not _led_hay_deck(scf):
        return
    try:
        scf.cf.param.set_value(f'{LED_GRUPO}.brightCorr', str(LED_BRILLO))
        scf.cf.param.set_value(f'{LED_GRUPO}.wrgb8888', '0')
        _led_estado['encendido'] = False
        print(f'✓ Color LED Deck listo (brillo {LED_BRILLO})')
    except Exception as e:
        print(f'Error inicializando LED: {e}')

def led_on(scf, color=None):
    if not LED_ACTIVO or scf is None or not _led_hay_deck(scf):
        return
    c = LED_COLOR if color is None else color
    try:
        scf.cf.param.set_value(f'{LED_GRUPO}.wrgb8888', str(int(c)))
        _led_estado['encendido'] = True
        print(f'💡 LED encendido (0x{int(c):08X})')
    except Exception as e:
        print(f'Error encendiendo LED: {e}')

def led_off(scf):
    if not LED_ACTIVO or scf is None:
        return
    if _led_estado['disponible'] is False:
        return
    try:
        scf.cf.param.set_value(f'{LED_GRUPO}.wrgb8888', '0')
        if _led_estado['encendido']:
            print('💡 LED apagado')
        _led_estado['encendido'] = False
    except Exception as e:
        print(f'Error apagando LED: {e}')

def interruptible_sleep(seconds, interval=0.05):
    elapsed = 0.0
    while elapsed < seconds:
        if stop_event.is_set():
            return False
        time.sleep(interval)
        elapsed += interval
    return True

# ═══════════════════════════════════════════════════════════════
# BLOQUE 3B — CÁMARA AI-DECK + MiDaS  (integrado de `Ai deck MiDas stream.py`)
# ═══════════════════════════════════════════════════════════════
#
# ARQUITECTURA (tres hilos, ninguno puede bloquear al paro de emergencia)
#
#   [AI-deck WiFi] → HiloReceptor ──cola(1)──> HiloProcesado ──_ultimo──> HiloVentana
#                    (solo socket)             (realce + MiDaS)           (imshow+tecla)
#
#   · HiloReceptor : lee el socket. Si se cae, avisa y termina; nada más.
#   · HiloProcesado: es el ÚNICO que llama a MiDaS. Si tarda 300 ms, solo se
#                    atrasa él; la cola es de tamaño 1 y descarta lo viejo.
#   · HiloVentana  : NO hace inferencia. Solo dibuja y llama a cv2.waitKey()
#                    cada ~30 ms. Por eso la tecla 'q' responde SIEMPRE, esté
#                    MiDaS ocupado, atascado o muerto.
#
#   Ninguno de los tres manda comandos de vuelo. La única acción sobre el
#   dron que pueden ejecutar es `activar_paro_emergencia`.

ANCHO_CAM  = 324          # Himax HM01B0 del ejemplo wifi-img-streamer
ALTO_CAM   = 244
MAGIC_CPX  = 0xBC         # byte mágico del header de imagen

# Los cuatro patrones posibles del mosaico Bayer. Si ves los colores
# intercambiados (cielo naranja, piel azulada), prueba otro en VISION_PATRON.
PATRONES_BAYER = {
    'BG': cv2.COLOR_BayerBG2BGR,
    'GB': cv2.COLOR_BayerGB2BGR,
    'RG': cv2.COLOR_BayerRG2BGR,
    'GR': cv2.COLOR_BayerGR2BGR,
} if _HAY_CV2 else {}


# Registro de cada caída del enlace: dónde estaba el dron y cuánto duró la
# sesión. Es lo que permite decidir CON DATOS si es distancia o es carga.
CAIDAS_WIFI = []

# ─────────────────────────────────────────────
# MONITOR DE BATERÍA (para cazar el brownout del ESP32)
# ─────────────────────────────────────────────
VBAT_LOG = []          # lista de (t, vbat, estado)
_bateria_estado = {'v': None, 't': 0.0, 'critica_desde': None}
_vbat_lock = threading.Lock()


class MonitorBateria:
    """
    Registra pm.vbat a 10 Hz mientras vuela.

    Para qué sirve: una LiPo de 1S en el Crazyflie va de ~4.2 V (llena) a
    ~3.0 V (vacía). En vuelo, cada acelerón de los motores la hunde de golpe.
    Si esos hundimientos coinciden con las caídas del WiFi, el ESP32 se está
    reiniciando por falta de tensión, y no hay ajuste de software que lo
    arregle: es batería, peso o consumo.

    No manda NADA al dron: solo se suscribe a un log. Si falla, se ignora.
    """

    def __init__(self):
        self.lg = None
        self.activo = False
        self.n = 0
        self._t_aviso = 0.0
        self._aviso_dado = False

    def iniciar(self, scf):
        if not MONITOR_BATERIA or scf is None:
            return False
        try:
            from cflib.crazyflie.log import LogConfig
            self.lg = LogConfig(name='bateria', period_in_ms=BATERIA_PERIODO_MS)
            self.lg.add_variable('pm.vbat', 'float')
            self.lg.add_variable('pm.state', 'int8_t')
            scf.cf.log.add_config(self.lg)
            self.lg.data_received_cb.add_callback(self._cb)
            self.lg.start()
            self.activo = True
            print(f'✓ Monitor de batería activo ({1000//BATERIA_PERIODO_MS} Hz)')
            return True
        except Exception as e:
            print(f'⚠ Monitor de batería no disponible: {e}')
            return False

    def _cb(self, timestamp, data, logconf):
        try:
            v = float(data.get('pm.vbat', 0.0))
            with _vbat_lock:
                VBAT_LOG.append((time.time(), v,
                                 int(data.get('pm.state', -1))))
                self.n += 1
                if len(VBAT_LOG) > 60000:       # ~100 min a 10 Hz
                    del VBAT_LOG[:10000]
            self._avisar(v)
        except Exception:
            pass

    def _avisar(self, v):
        """
        Avisa EN VUELO de que la batería está baja.

        Esto faltaba: en el vuelo del 2026-08-22 la tensión estuvo entre 2.74
        y 3.15 V de principio a fin y nadie se enteró hasta el informe final.
        Con eso el ESP32 se reseteaba (10 atascos del stream) y la LiPo se
        estaba dañando. Ahora se ve mientras pasa.
        """
        if v <= 0.1:
            return
        ahora = time.time()
        _bateria_estado['v'] = v
        _bateria_estado['t'] = ahora
        if v < BATERIA_CRITICA_V:
            if _bateria_estado['critica_desde'] is None:
                _bateria_estado['critica_desde'] = ahora
        else:
            _bateria_estado['critica_desde'] = None
        if v < BATERIA_CRITICA_V:
            if ahora - self._t_aviso > 5.0:
                self._t_aviso = ahora
                print(f'🔋 BATERIA CRITICA: {v:.2f} V — por debajo de '
                      f'{BATERIA_CRITICA_V:.2f} V. ATERRIZA Y CAMBIALA.')
                print('   A esta tension la LiPo se daña y el ESP32 del '
                      'AI-deck se resetea.')
        elif v < BATERIA_AVISO_V and not self._aviso_dado:
            self._aviso_dado = True
            print(f'🔋 Bateria baja: {v:.2f} V (aviso a partir de '
                  f'{BATERIA_AVISO_V:.2f} V). Te quedan pocos vuelos.')

    def detener(self):
        try:
            if self.lg and self.activo:
                self.lg.stop()
        except Exception:
            pass
        self.activo = False


def comprobar_bateria_para_volar(espera=2.0):
    """
    Lee la tensión antes de despegar. Devuelve True si se puede volar.

    Por qué existe: por debajo de ~3.0 V en vuelo el Crazyflie no tiene
    empuje para seguir la ruta —oscila y se va— y ninguna medida de error
    de seguimiento significa nada. Además cada vuelo así daña la LiPo de
    forma permanente. Es más útil no despegar que volar y descartar datos.
    """
    if not MONITOR_BATERIA:
        print('🔋 (Monitor de batería apagado: no se comprueba la tensión. '
              'Asegúrate a mano de que esté cargada.)')
        return True

    t0 = time.time()
    while time.time() - t0 < espera:
        if vbat_ahora() is not None:
            break
        time.sleep(0.1)

    v = vbat_ahora()
    if v is None:
        print('⚠ No se pudo leer la tensión de la batería; se vuela sin comprobar.')
        return True

    print(f'🔋 Bateria antes de despegar: {v:.2f} V')
    if v >= BATERIA_MINIMA_DESPEGUE:
        return True

    print()
    print('=' * 62)
    print(f'  VUELO BLOQUEADO — BATERIA A {v:.2f} V')
    print('=' * 62)
    print(f'  Minimo para despegar: {BATERIA_MINIMA_DESPEGUE:.2f} V')
    print()
    print('  Una LiPo 1S cargada marca 4.0-4.2 V en reposo. A esta tension:')
    print('   · los motores no dan empuje para seguir la ruta (oscila),')
    print('   · el GAP8 del AI-deck se cuelga y la camara se congela,')
    print('   · y cada vuelo daña la bateria un poco mas, sin vuelta atras.')
    print()
    print('  Carga la bateria o pon una cargada. Si de verdad quieres volar')
    print('  asi, pon PERMITIR_VOLAR_SIN_BATERIA = True arriba del archivo.')
    print('=' * 62)
    return bool(PERMITIR_VOLAR_SIN_BATERIA)


def informe_seguimiento_ventana(ruta_2d):
    """
    Error de seguimiento medido SOLO durante el recorrido de la ruta.

    El informe original incluye todo lo que haya en `real_trajectory`:
    despegue, aproximación al inicio y aterrizaje. Con la aproximación
    midiendo más de un metro sobre una ruta de doce, y con `error_seguimiento`
    anclando cada muestra por longitud de arco recorrida, ese metro de más
    desplaza TODAS las muestras respecto a la ruta. En una ruta de cobertura,
    donde los surcos están a 20 cm, un desplazamiento así empareja cada punto
    con el surco equivocado y el error sale muy por encima del real.

    Aquí se recorta a la ventana exacta entre `start_trajectory` y su final.
    """
    ini, fin = VENTANA_RUTA['ini'], VENTANA_RUTA['fin']
    if ini is None or fin is None or len(tiempos_trayectoria) < 50:
        return
    t = np.asarray(tiempos_trayectoria, dtype=float)
    n = min(len(t), len(real_trajectory))
    t = t[:n]
    xyz = np.asarray(real_trajectory[:n], dtype=float)

    dentro = (t >= ini) & (t <= fin)
    if dentro.sum() < 50:
        return
    vuelo = xyz[dentro]

    errs, _ = error_seguimiento(vuelo[:, :2], ruta_2d)
    rms = float(np.sqrt(np.mean(errs ** 2)))
    p95 = float(np.percentile(errs, 95))

    print()
    print('─' * 62)
    print('  ERROR SOLO DURANTE LA RUTA  (sin despegue ni aproximación)')
    print('─' * 62)
    print(f'  Muestras              : {int(dentro.sum())} de {n}')
    print(f'  Ventana               : {fin-ini:.1f} s')
    print(f'  RMS                   : {rms*100:6.2f} cm')
    print(f'  Percentil 95          : {p95*100:6.2f} cm')
    print(f'  Máximo                : {errs.max()*100:6.2f} cm')
    print()
    print('  Este es el número honesto. El de abajo incluye la aproximación')
    print('  al inicio, que no es parte de la ruta y desplaza el anclaje.')
    print('─' * 62)


def yaw_desde_cuaternion(ori):
    """Yaw (rad) de un cuaternión {x,y,z,w}. None si no hay orientación."""
    if not ori:
        return None
    try:
        x, y, z, w = (float(ori['x']), float(ori['y']),
                      float(ori['z']), float(ori['w']))
        return float(np.arctan2(2.0 * (w * z + x * y),
                                1.0 - 2.0 * (y * y + z * z)))
    except Exception:
        return None


class MonitorEstimador:
    """
    Escucha stateEstimate.{x,y,z,yaw}: la posición que el dron CREE tener.

    No manda nada, solo se suscribe a un log. Comparándolo con el MoCap se
    ve de un vistazo si el problema es que el dron no sabe dónde está o que
    no logra seguir la referencia.
    """

    def __init__(self):
        self.lg = None
        self.activo = False
        # Columnas:
        #  0 t
        #  1..3 ekf x,y,z      4 ekf yaw (°)
        #  5..7 mocap FILTRADO x,y,z   (lo que se le manda al dron)
        #  8    mocap yaw (°)
        #  9..11 mocap CRUDO x,y,z     (dónde estaba el dron de verdad)
        self.datos = []

    def iniciar(self, scf):
        if not MONITOR_ESTIMADOR or scf is None:
            return False
        try:
            from cflib.crazyflie.log import LogConfig
            self.lg = LogConfig(name='est', period_in_ms=ESTIMADOR_PERIODO_MS)
            self.lg.add_variable('stateEstimate.x', 'float')
            self.lg.add_variable('stateEstimate.y', 'float')
            self.lg.add_variable('stateEstimate.z', 'float')
            self.lg.add_variable('stateEstimate.yaw', 'float')
            scf.cf.log.add_config(self.lg)
            self.lg.data_received_cb.add_callback(self._cb)
            self.lg.start()
            self.activo = True
            print(f'✓ Monitor del estimador activo '
                  f'({1000//ESTIMADOR_PERIODO_MS} Hz)')
            return True
        except Exception as e:
            print(f'⚠ Monitor del estimador no disponible: {e}')
            return False

    def _cb(self, timestamp, data, logconf):
        try:
            ymoc = yaw_desde_cuaternion(_mocap_ori)
            self.datos.append((
                time.time(),
                float(data.get('stateEstimate.x', 0.0)),
                float(data.get('stateEstimate.y', 0.0)),
                float(data.get('stateEstimate.z', 0.0)),
                float(data.get('stateEstimate.yaw', 0.0)),   # grados
                mocap_pose['x'], mocap_pose['y'], mocap_pose['z'],
                np.degrees(ymoc) if ymoc is not None else np.nan,
                mocap_pose_cruda['x'], mocap_pose_cruda['y'],
                mocap_pose_cruda['z'],
            ))
        except Exception:
            pass

    def detener(self):
        try:
            if self.lg and self.activo:
                self.lg.stop()
        except Exception:
            pass
        self.activo = False

    def informe(self, altura_vuelo=0.3):
        """Separa el error total en estimación, yaw y control."""
        if len(self.datos) < 20:
            print('\n[est] Sin datos suficientes del estimador.')
            return
        d = np.array(self.datos, dtype=float)
        en_vuelo = d[:, 3] > 0.6 * altura_vuelo
        if en_vuelo.sum() < 20:
            en_vuelo = np.ones(len(d), dtype=bool)
        d = d[en_vuelo]

        ekf = d[:, 1:3]
        moc = d[:, 5:7]                       # MoCap FILTRADO (escalera)
        crudo_ok = d.shape[1] >= 12 and np.any(d[:, 9:11] != 0.0)
        cru = d[:, 9:11] if crudo_ok else moc  # MoCap CRUDO (verdad)

        err       = np.linalg.norm(ekf - cru, axis=1)   # error REAL
        err_esc   = np.linalg.norm(ekf - moc, axis=1)   # contra la escalera
        # Cuánto se mueve el dron entre dos correcciones: esa distancia es
        # el error FALSO que introduce la escalera, y no dice nada del EKF.
        salto_escalera = np.linalg.norm(cru - moc, axis=1)

        print()
        print('─' * 62)
        print('  ¿DÓNDE SE PIERDE LA PRECISIÓN?')
        print('─' * 62)
        print(f'  Muestras en vuelo    : {len(d)}')
        print()
        print('  1) ESTIMACIÓN DE POSICIÓN  (lo que cree vs. dónde está)')
        if crudo_ok:
            print(f'     RMS               : {err.mean()*100:6.1f} cm'
                  f'   ← contra el MoCap real')
            print(f'     Máximo            : {err.max()*100:6.1f} cm')
            # Cuánto INFLABA de verdad la medida vieja. Se calcula restando,
            # no suponiendo: si el hueco de la escalera es grande pero el
            # error apenas cambia, es que el error es del EKF y no del hueco,
            # y decir lo contrario sería mentir con dos cifras al lado.
            inflado = err_esc.mean() - err.mean()
            if abs(inflado) > 0.02:
                print()
                print(f'     [contra el MoCap filtrado: {err_esc.mean()*100:.1f} cm '
                      f'→ {inflado*100:+.1f} cm]')
                print( '     Esa diferencia era el escalón, no el estimador.')
            elif salto_escalera.mean() > 0.02:
                print()
                print(f'     El MoCap se congela {salto_escalera.mean()*100:.1f} cm '
                      f'entre correcciones, pero medir')
                print(f'     contra él da casi lo mismo '
                      f'({err_esc.mean()*100:.1f} cm). O sea que estos '
                      f'{err.mean()*100:.1f} cm')
                print( '     son error REAL del estimador, no un artefacto de')
                print( '     la medida. El dron de verdad no sabe dónde está.')
        else:
            print(f'     RMS               : {err.mean()*100:6.1f} cm')
            print(f'     Máximo            : {err.max()*100:6.1f} cm')
            print( '     (sin pose cruda registrada: la cifra puede estar')
            print( '      inflada por la cadencia baja del MoCap)')
        estimacion_ok = err.mean() < 0.15

        # ── 2) YAW: la pieza que decide HACIA DÓNDE empuja ──
        yaw_ekf = d[:, 4]
        yaw_moc = d[:, 8]
        hay_yaw = np.isfinite(yaw_moc).sum() > 20
        print()
        print('  2) YAW  (decide en qué dirección empuja el controlador)')
        if hay_yaw:
            m = np.isfinite(yaw_moc)
            dy = (yaw_ekf[m] - yaw_moc[m] + 180.0) % 360.0 - 180.0
            dy = dy - np.median(dy)      # quita el offset fijo de montaje
            print(f'     Error de yaw RMS  : {np.sqrt(np.mean(dy**2)):6.1f} °')
            print(f'     Máximo            : {np.abs(dy).max():6.1f} °')
            print(f'     Deriva (ini→fin)  : {dy[-1]-dy[0]:+6.1f} °')
            yaw_malo = np.sqrt(np.mean(dy ** 2)) > 12.0
        else:
            print('     No se puede medir: el mensaje del MoCap no trae')
            print('     orientación. Comprueba con `python diagnostico.py mocap`.')
            yaw_malo = None

        # ── 3) LATENCIA de la cadena MoCap → EKF ──
        # Se busca el desfase temporal que mejor superpone las dos señales.
        #
        # CUIDADO CON ESTA MEDIDA. Solo tiene sentido contra la pose CRUDA.
        # Contra la filtrada da un número que parece latencia pero no lo es:
        # una señal muestreada a 1 Hz y retenida está, por construcción, medio
        # segundo vieja de media, y el ajuste devuelve ese retardo de muestreo
        # aunque la red no tarde absolutamente nada. Simulado con latencia
        # CERO, muestrear a 1 Hz ya devuelve −180 ms.
        #
        # Convenio de signo (verificado inyectando un retardo conocido):
        # el desfase sale NEGATIVO cuando el MoCap va atrasado, y su valor
        # absoluto es el atraso.
        lat = None
        medida_fiable = crudo_ok
        try:
            t = d[:, 0] - d[0, 0]
            ref_x = d[:, 9] if crudo_ok else d[:, 5]
            ref_y = d[:, 10] if crudo_ok else d[:, 6]
            if t[-1] > 5.0 and len(t) > 100:
                mejor, mejor_e = None, 1e9
                for lag in np.arange(-0.5, 1.21, 0.01):
                    mx = np.interp(t - lag, t, ref_x)
                    my = np.interp(t - lag, t, ref_y)
                    e = np.mean((d[:, 1] - mx) ** 2 + (d[:, 2] - my) ** 2)
                    if e < mejor_e:
                        mejor_e, mejor = e, lag
                lat = float(mejor)
        except Exception:
            pass

        print()
        print('  3) LATENCIA de la cadena MoCap → EKF')
        if lat is None:
            print('     No se pudo estimar (vuelo demasiado corto).')
        elif not medida_fiable:
            print('     NO MEDIBLE en este vuelo.')
            print('     Solo hay pose filtrada, y a cadencia baja el ajuste')
            print('     devuelve el retardo del propio muestreo, no el de la')
            print('     red. Vuelve a volar con esta versión del script, que')
            print('     ya registra la pose sin filtrar.')
        else:
            atraso = abs(lat)
            print(f'     Desfase medido    : {lat*1000:+6.0f} ms')
            if atraso < 0.03:
                print('     La cadena va prácticamente al día. El retraso de')
                print('     red NO es lo que limita la precisión.')
            else:
                print(f'     El dato del MoCap llega ~{atraso*1000:.0f} ms viejo')
                print( '     (Motive → Node-RED → MQTT → Python → radio).')
                # El periodo de corrección debe superar varias veces el
                # retraso, o el lazo realimenta posiciones viejas y oscila.
                f_seg = 1.0 / max(atraso * 6.0, 0.02)
                print(f'     Cadencia segura   : hasta ~{f_seg:4.1f} Hz')
                print( '     (regla: el periodo de corrección por encima de')
                print( '      unas 6 veces el retraso.)')

        # ── 4) CUÁNTO VUELA A CIEGAS ──
        # Entre corrección y corrección el EKF navega solo con la IMU. Esa
        # distancia es lo que puede desviarse sin que nadie lo corrija, y es
        # el techo de precisión de la ruta: por muy buena que sea la
        # trayectoria, el dron no puede seguirla mejor que esto.
        print()
        print('  4) VUELO A CIEGAS entre correcciones')
        try:
            dt_log = np.median(np.diff(d[:, 0]))
            cambios = np.linalg.norm(np.diff(moc, axis=0), axis=1) > 1e-6
            n_corr = int(cambios.sum())
            dur = d[-1, 0] - d[0, 0]
            hz_corr = n_corr / dur if dur > 0 else 0.0
            vel = (np.linalg.norm(np.diff(cru, axis=0), axis=1) / dt_log
                   if crudo_ok else np.array([0.0]))
            v_med = float(np.median(vel))
            hz_log = 1.0 / dt_log if dt_log > 0 else 0.0
            satura = hz_corr > 0.9 * hz_log
            print(f'     Correcciones al dron : {hz_corr:5.2f} Hz '
                  f'({n_corr} en {dur:.0f} s)'
                  + ('  [tope del monitor]' if satura else ''))
            if satura:
                print(f'     El monitor solo mira {hz_log:.0f} veces por segundo,')
                print( '     así que la cadencia real puede ser mayor. Para el')
                print( '     dato exacto: `python diagnostico.py mocap`.')
            if crudo_ok and hz_corr > 0:
                hueco = 1.0 / hz_corr
                print(f'     Velocidad media      : {v_med:5.2f} m/s')
                print(f'     Recorre a ciegas     : {v_med*hueco*100:5.1f} cm '
                      f'entre corrección y corrección')
                print()
                print( '     Ese recorrido es el margen dentro del cual el dron')
                print( '     no tiene ni idea de si se está yendo de la ruta.')
        except Exception:
            print('     (no se pudo calcular)')

        print()
        print('  ─ INTERPRETACIÓN ─')
        if not estimacion_ok:
            print('  → El dron NO SABE DÓNDE ESTÁ. Sube la cadencia del MoCap.')
        elif yaw_malo:
            print('  → LA POSICIÓN ESTÁ BIEN PERO EL YAW ESTÁ DESVIADO.')
            print('    El controlador convierte el error de posición (en el')
            print('    mundo) a roll/pitch (en el cuerpo) usando el yaw')
            print('    ESTIMADO. Si ese yaw está torcido, empuja torcido: no')
            print('    corrige hacia la ruta sino a un lado, se pasa, vuelve a')
            print('    corregir torcido, y describe lazos que nunca cierran.')
            print('    Un error de yaw de 20° convierte un regulador estable')
            print('    en una espiral. Es exactamente lo que se ve en la')
            print('    gráfica: ni una recta, ni una curva limpia.')
            print()
            print('    ARREGLO: USAR_ORIENTACION_MOCAP = True')
            print('    PRUEBA RÁPIDA: YAW_ACTIVO = False. Con la nariz fija el')
            print('    giroscopio casi no deriva. Si con eso la ruta sale')
            print('    limpia, el yaw queda confirmado como culpable.')
        elif yaw_malo is None:
            print('  → La posición estimada es buena, así que el error es de')
            print('    CONTROL. El sospechoso número uno es el YAW, y no se')
            print('    puede medir sin la orientación del MoCap.')
            print()
            print('    PRUEBA DECISIVA Y GRATIS: pon YAW_ACTIVO = False y')
            print('    repite el vuelo. Con la nariz fija el giroscopio apenas')
            print('    deriva, así que si el error se desploma, era el yaw.')
            print('    Si el error sigue igual, el yaw queda descartado y hay')
            print('    que mirar el controlador de posición.')
        else:
            print('  → Estimación y yaw correctos. El error restante es de')
            print('    seguimiento puro: baja TRAY_V_CRUCERO y TRAY_A_LAT_MAX.')

        if medida_fiable and lat is not None and abs(lat) > 0.15:
            print()
            print(f'  ⚠ LATENCIA ALTA ({abs(lat)*1000:.0f} ms) y esta vez sí está')
            print('    medida contra la pose sin filtrar, así que es real.')
            print('    Antes de subir la cadencia del MoCap hay que bajarla,')
            print(f'    o subir extPosStdDev (ahora {EXT_POS_STD_DEV}) para que')
            print('    el EKF se fíe menos de cada corrección.')
        print('─' * 62)


def vbat_ahora():
    """Última tensión medida, o None."""
    with _vbat_lock:
        return VBAT_LOG[-1][1] if VBAT_LOG else None


def vbat_minimo(t_fin, ventana=BATERIA_VENTANA_S):
    """Tensión MÍNIMA en los `ventana` segundos anteriores a t_fin."""
    with _vbat_lock:
        vals = [v for (t, v, _s) in VBAT_LOG if t_fin - ventana <= t <= t_fin]
    return min(vals) if vals else None


@dataclass
class FrameCam:
    """Un frame decodificado + metadatos."""
    imagen: np.ndarray      # BGR uint8, listo para OpenCV
    timestamp: float
    formato: int            # 0 = raw, 1 = JPEG
    monotonic_ts: float = 0.0  # inicio de recepción; conserva edad en las colas


_timer_fino = {'activo': False}


def afinar_temporizador_windows(activar=True):
    """
    Windows duerme en saltos de ~15.6 ms por defecto. Con eso, un
    `time.sleep(0.02)` para 50 Hz se convierte en ~31 ms reales → ~32 Hz, y
    medido sale aún menos. timeBeginPeriod(1) baja el salto a 1 ms y permite
    cumplir de verdad la cadencia pedida.

    Es exactamente por esto que el hilo de extpos daba ~22 Hz pidiendo 50,
    con 0.0 ms por envío: la radio no era el problema, era el reloj.
    """
    try:
        import ctypes
        if activar and not _timer_fino['activo']:
            ctypes.windll.winmm.timeBeginPeriod(1)
            _timer_fino['activo'] = True
        elif not activar and _timer_fino['activo']:
            ctypes.windll.winmm.timeEndPeriod(1)
            _timer_fino['activo'] = False
        return True
    except Exception:
        return False


def _bajar_prioridad_hilo():
    """
    Baja la prioridad del hilo actual (solo Windows). Así, si la CPU se
    satura, Windows le quita tiempo a la visión antes que al MoCap o al
    teclado. Si falla, no pasa nada.
    """
    if not VISION_PRIORIDAD_BAJA:
        return
    try:
        import ctypes
        THREAD_PRIORITY_BELOW_NORMAL = -1
        h = ctypes.windll.kernel32.GetCurrentThread()
        ctypes.windll.kernel32.SetThreadPriority(h, THREAD_PRIORITY_BELOW_NORMAL)
    except Exception:
        pass


# ── 1) Receptor del stream (protocolo CPX del ejemplo wifi-img-streamer) ──
class HiloReceptorAIDeck(threading.Thread):
    """
    Estructura de cada paquete:
        [uint16 length][uint8 routing/dst][uint8 function/src]  <- 4 bytes
        payload de (length - 2) bytes

    El primer paquete de cada imagen trae el header:
        <BHHBBI  ->  magic, width, height, depth, format, size
    Y luego vienen N paquetes de datos hasta completar `size` bytes.
    """

    def __init__(self, ip, puerto, cola, bayer=True, patron='BG', demosaico=0):
        super().__init__(daemon=True, name='rx-aideck')
        self.ip = ip
        self.puerto = puerto
        self.cola = cola
        self.bayer = bayer
        self.patron = patron
        self.demosaico = demosaico
        self._detener = threading.Event()
        self.sock = None
        self.frames_recibidos = 0
        self.conectado = False
        self.error = None
        self.caidas = 0          # nº de veces que se ha caído el enlace
        self.atascos = 0         # nº de veces que el stream se quedó mudo
        self.t_ultimo_frame = None
        self.bytes_totales = 0
        self.t_inicio = None
        self.formatos = {0: 0, 1: 0}
        self.sesiones_vacias = 0   # reconexiones seguidas con CERO frames
        self._t_sesion = None
        self._frames_sesion = 0
        self.atascos_lentos = 0    # reconexiones forzadas por caudal bajo
        self.fps_enlace = 0.0      # fps medidos en la ventana del vigilante
        self.fps_min_visto = None  # el peor caudal de todo el vuelo
        self.proc = None           # puente_camara.py, si VISION_RECEPTOR_PROCESO
        self._proceso_ok = True
        self._motivo_proceso = ''
        self._v_t0 = 0.0
        self._v_n0 = 0

    # -- utilidades de socket ------------------------------------------
    def _recibir_exacto(self, n):
        datos = bytearray()
        while len(datos) < n:
            if self._detener.is_set() or stop_event.is_set():
                raise ConnectionAbortedError('Detenido')
            try:
                trozo = self.sock.recv(n - len(datos))
                self.bytes_totales += len(trozo)
            except socket.timeout:
                raise TimeoutError(
                    f'sin datos en {VISION_TIMEOUT_LECTURA:.0f}s '
                    '(GAP8 atascado: el socket sigue abierto pero no llegan '
                    'frames)')
            if not trozo:
                raise ConnectionResetError('El AI-deck cerró la conexión')
            datos.extend(trozo)
        return datos

    def _leer_paquete(self):
        cabecera = self._recibir_exacto(4)
        length, _dst, _src = struct.unpack('<HBB', cabecera)
        return self._recibir_exacto(length - 2)

    # -- decodificación -------------------------------------------------
    def _decodificar(self, buffer, formato, ancho, alto):
        if formato == 0:
            # RAW, 1 byte por píxel. Con la cámara Bayer hay que interpolar.
            arr = np.frombuffer(buffer, dtype=np.uint8)
            if arr.size < ancho * alto:
                return None
            arr = arr[: ancho * alto].reshape((alto, ancho))
            if self.bayer:
                tabla = DEMOSAICO.get(self.demosaico, DEMOSAICO[0])[1]
                return cv2.cvtColor(arr, tabla[self.patron])
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        # JPEG comprimido por el GAP8
        arr = np.frombuffer(buffer, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            if self.bayer and VISION_JPEG_BAYER_GRIS:
                tabla = DEMOSAICO.get(self.demosaico, DEMOSAICO[0])[1]
                img = cv2.cvtColor(img, tabla[self.patron])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    # -- registro de caídas ----------------------------------------------
    def _registrar_caida(self, motivo, tipo='caida'):
        """Guarda DÓNDE y CUÁNDO se cayó el enlace. Nunca lanza excepción."""
        try:
            ahora = time.time()
            CAIDAS_WIFI.append({
                'n'             : self.caidas,
                't'             : ahora,
                'dur_sesion_s'  : (ahora - self._t_sesion) if self._t_sesion else 0.0,
                'frames_sesion' : self.frames_recibidos - self._frames_sesion,
                'pose'          : dict(mocap_pose),
                'vbat'          : vbat_ahora(),
                'vbat_min'      : vbat_minimo(ahora),
                'tipo'          : tipo,          # 'atasco' | 'caida'
                'motivo'        : motivo[:120],
            })
        except Exception:
            pass

    # -- sesión a través de puente_camara.py, en un proceso aparte ---------
    def _sesion_proceso(self):
        """
        Lo mismo que `_sesion`, pero el socket lo lee puente_camara.py en su
        propio proceso (su propio GIL). Aquí solo llegan imágenes completas:
        una lectura por imagen en vez de ~160, así que la radio, el MoCap y
        el control ya no la atascan. Ver la cabecera de puente_camara.py.
        """
        import subprocess
        cab = struct.Struct('<IHHBdQ')
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'puente_camara.py')
        self.proc = subprocess.Popen(
            [sys.executable, ruta, self.ip, str(self.puerto),
             str(VISION_TIMEOUT_LECTURA)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        tubo = self.proc.stdout
        base = self.bytes_totales
        try:
            while not self._detener.is_set() and not stop_event.is_set():
                c = tubo.read(cab.size)
                if len(c) < cab.size:
                    break
                tam, ancho, alto, formato, t_cap, bytes_red = cab.unpack(c)
                self.bytes_totales = base + bytes_red
                if formato == 255:
                    self.conectado = True
                    self._t_sesion = time.time()
                    self._frames_sesion = self.frames_recibidos
                    if self.t_inicio is None:
                        self.t_inicio = time.time()
                    self._v_t0 = self._t_sesion
                    self._v_n0 = self.frames_recibidos
                    print('[cam] Conectado (receptor en proceso aparte). '
                          'Recibiendo imágenes.')
                    continue
                cuerpo = tubo.read(tam)
                if len(cuerpo) < tam:
                    break
                self._entregar(cuerpo, formato, ancho, alto, t_cap)
        finally:
            self._cerrar_proceso()
        if self._detener.is_set() or stop_event.is_set():
            return
        # El puente terminó por su cuenta: traducir su motivo a las mismas
        # excepciones que el modo directo, para que `run` lo registre igual.
        motivo = (self._motivo_proceso or '').strip()
        if motivo.startswith('ATASCO'):
            raise TimeoutError(motivo[len('ATASCO '):] or 'sin datos')
        raise ConnectionResetError(motivo[len('CAIDA '):] if motivo.startswith('CAIDA')
                                   else (motivo or 'el puente de la cámara terminó'))

    def _cerrar_proceso(self):
        proc, self.proc = self.proc, None
        self._motivo_proceso = ''
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=2.0)
            self._motivo_proceso = proc.stderr.read().decode('utf-8', 'replace')
        except Exception:
            pass

    # -- una sesión de streaming (desde connect hasta que se cae) ---------
    def _sesion(self):
        if VISION_RECEPTOR_PROCESO and self._proceso_ok:
            try:
                return self._sesion_proceso()
            except (FileNotFoundError, PermissionError) as e:
                # Sin el puente (archivo o intérprete): modo directo de siempre.
                self._proceso_ok = False
                print(f'[cam] ⚠ No se pudo lanzar puente_camara.py ({e}); '
                      'receptor en este proceso.')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)
        self.sock.connect((self.ip, self.puerto))
        # Timeout de LECTURA, no infinito. Antes esto era settimeout(None):
        # si el GAP8 se atascaba pero el ESP32 mantenía el socket abierto,
        # recv() se quedaba bloqueado PARA SIEMPRE. La imagen se congelaba,
        # no saltaba ningún error y el contador de caídas seguía en 0, que
        # es justo el síntoma imposible de diagnosticar.
        self.sock.settimeout(VISION_TIMEOUT_LECTURA)
        # TCP_NODELAY: los frames vienen troceados; sin esto Nagle añade
        # latencia y agrava los cortes bajo carga.
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        try:
            # 1 MB de buffer de recepción (por defecto suele ser 64 kB). Una
            # imagen RAW son 79 kB: con el buffer pequeño, un parón de unas
            # décimas en este hilo basta para llenarlo y frenar al GAP8.
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        except OSError:
            pass
        self.conectado = True
        self._t_sesion = time.time()
        self._frames_sesion = self.frames_recibidos
        # Reloj desde la CONEXIÓN, no desde el primer frame: si no llega
        # ninguno, el kB/s debe salir 0, no un número inventado por dividir
        # entre una fracción de segundo.
        if self.t_inicio is None:
            self.t_inicio = time.time()
        print('[cam] Conectado. Recibiendo imágenes.')

        # Ventana deslizante del vigilante de caudal. Deslizante y no
        # acumulada a propósito: si el enlace arranca bien y se degrada a
        # mitad de vuelo, una media desde el principio tarda minutos en
        # bajar del umbral, y para entonces el vuelo ya se perdió.
        self._v_t0 = self._t_sesion
        self._v_n0 = self.frames_recibidos

        while not self._detener.is_set() and not stop_event.is_set():
            payload = self._leer_paquete()
            if len(payload) < 11:
                continue
            magic, ancho, alto, _prof, formato, tam = struct.unpack('<BHHBBI', payload[:11])
            if magic != MAGIC_CPX:
                continue

            captura_monotonica = time.perf_counter()
            cuerpo = bytearray()
            while len(cuerpo) < tam:
                cuerpo.extend(self._leer_paquete())

            self._entregar(bytes(cuerpo), formato, ancho, alto, captura_monotonica)

    def _entregar(self, cuerpo, formato, ancho, alto, captura_monotonica):
        """Decodifica, publica en la cola de 1 y vigila el caudal."""
        img = self._decodificar(cuerpo, formato, ancho, alto)
        if img is None:
            return

        self.frames_recibidos += 1
        self.formatos[formato] = self.formatos.get(formato, 0) + 1
        self.t_ultimo_frame = time.time()
        fr = FrameCam(imagen=img, timestamp=self.t_ultimo_frame,
                      formato=formato, monotonic_ts=captura_monotonica)

        # Cola de 1: si el procesamiento se atrasa, tiramos el frame
        # viejo y nos quedamos con el más reciente. Nunca se bloquea.
        try:
            self.cola.put_nowait(fr)
        except queue.Full:
            try:
                self.cola.get_nowait()
            except queue.Empty:
                pass
            try:
                self.cola.put_nowait(fr)
            except queue.Full:
                pass

        # ── Vigilante de CAUDAL ──
        # Se evalúa al llegar cada frame, no en un hilo aparte: así no
        # hay un hilo más compitiendo, y un stream tan lento que ni
        # llegue a completar la ventana ya lo caza el timeout de lectura.
        if VISION_FPS_MINIMO > 0 and self.atascos_lentos < VISION_VIGILANTE_MAX:
            dt_v = self.t_ultimo_frame - self._v_t0
            if dt_v >= VISION_VIGILANTE_GRACIA_S:
                fps_v = (self.frames_recibidos - self._v_n0) / dt_v
                self.fps_enlace = fps_v
                if self.fps_min_visto is None or fps_v < self.fps_min_visto:
                    self.fps_min_visto = fps_v
                self._v_t0, self._v_n0 = self.t_ultimo_frame, self.frames_recibidos
                if fps_v < VISION_FPS_MINIMO:
                    self.atascos_lentos += 1
                    # TimeoutError hereda de OSError, así que lo recoge el
                    # mismo `except` que las caídas y queda marcado como
                    # 'atasco' en el informe, que es lo que es.
                    raise TimeoutError(
                        f'caudal de {fps_v:.2f} fps durante {dt_v:.0f} s '
                        f'(el GAP8 manda, pero a cuentagotas; lo normal '
                        f'en vuelo son ~5 fps)')

    # -- bucle principal, con reconexión ---------------------------------
    def run(self):
        # OJO: aquí NO se baja la prioridad, y es a propósito.
        # Antes sí se bajaba, y fue un error mío: este hilo casi no gasta CPU
        # (solo vacía un socket), pero si el sistema no lo despacha a tiempo
        # deja de drenar los ~480 kB/s de vídeo RAW. Entonces se llena la
        # ventana TCP, el GAP8 se queda bloqueado escribiendo, y desde aquí
        # se ve como "el GAP8 no manda nada" — cuando en realidad el que no
        # estaba leyendo era la PC.
        # Encaja con la evidencia: en el banco (PC libre) 0 atascos en 9
        # minutos; en vuelo (PC con MQTT, radio, MiDaS y matplotlib) atascos
        # constantes. Lo que sí conviene degradar es el hilo de MiDaS.
        intento = 0
        primera = True

        while not self._detener.is_set() and not stop_event.is_set():
            try:
                if primera:
                    print(f'[cam] Conectando al AI-deck {self.ip}:{self.puerto} ...')
                    primera = False
                else:
                    print(f'[cam] Reintentando conexión (intento {intento})...')
                self._sesion()
                # Salida limpia (paro o detener): no reintentar.
                break

            except (ConnectionAbortedError, ConnectionResetError, OSError) as e:
                # Un cierre PEDIDO por nosotros (fin de vuelo, paro de
                # emergencia) no es una caída del enlace. Contarlo ensuciaba
                # el informe con un falso positivo en la pose (0,0,0).
                if self._detener.is_set() or stop_event.is_set():
                    break
                self.error = str(e)
                self.caidas += 1
                self._registrar_caida(str(e))
                if isinstance(e, TimeoutError):
                    self.atascos += 1
                    CAIDAS_WIFI[-1]['tipo'] = 'atasco'
                    CAIDAS_WIFI[-1]['lento'] = 'caudal de' in str(e)
                    print(f'[cam] ⚠ Stream ATASCADO: {e}')
                    if self.atascos_lentos >= VISION_VIGILANTE_MAX:
                        print(f'[cam]   Van {self.atascos_lentos} reconexiones '
                              'por caudal bajo y no ha servido: el GAP8 está '
                              'colgado, y de eso no se sale por software.')
                        print('[cam]   QUITA Y PON LA BATERÍA del dron. Dejo de '
                              'vigilar el caudal para no reconectar en bucle; '
                              'los fps siguen saliendo en la ventana.')
                    else:
                        print('[cam]   Reconectando. Si se repite, es el GAP8, '
                              'no el WiFi.')
                else:
                    print(f'[cam] ⚠ Enlace con el AI-deck caído: {e}')
            except Exception as e:
                if self._detener.is_set() or stop_event.is_set():
                    break
                self.error = str(e)
                self.caidas += 1
                self._registrar_caida(str(e))
                print(f'[cam] ⚠ Error en el receptor: {e}')
            finally:
                self.conectado = False
                if self.sock:
                    # shutdown ANTES de close: así el ESP32 (y con suerte el
                    # GAP8) ven una desconexión ordenada y liberan su estado.
                    # Un close a secas puede dejar al GAP8 creyendo que aún
                    # hay cliente y no reanudar el stream nunca.
                    try:
                        self.sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    try:
                        self.sock.close()
                    except OSError:
                        pass
                self.sock = None

            # ── ¿GAP8 colgado? ──
            # Si varias reconexiones seguidas no traen ni un frame, el GAP8
            # está colgado y reconectar NO lo recupera: el ESP32 acepta la
            # conexión pero nadie está capturando al otro lado. Seguir
            # reintentando solo llena la consola durante el vuelo.
            if CAIDAS_WIFI and CAIDAS_WIFI[-1].get('frames_sesion', 0) == 0:
                self.sesiones_vacias += 1
            else:
                self.sesiones_vacias = 0

            if self.sesiones_vacias >= VISION_MAX_SESIONES_VACIAS:
                print('[cam] ═══ GAP8 COLGADO ═══')
                print(f'[cam] {self.sesiones_vacias} reconexiones seguidas sin un '
                      'solo frame. Reconectar no lo va a arreglar:')
                print('[cam] el ESP32 acepta la conexión pero el GAP8 no captura.')
                print('[cam] APAGA Y ENCIENDE EL DRON para recuperarlo (un reset')
                print('[cam] del ESP32 solo no basta). Dejo de reintentar para no')
                print('[cam] llenar la consola; el vuelo sigue sin cámara.')
                break

            if self._detener.is_set() or stop_event.is_set():
                break
            if not VISION_RECONECTAR:
                print('[cam]   Reconexión desactivada. El vuelo continúa SIN cámara.')
                break

            intento += 1
            if VISION_MAX_REINTENTOS and intento > VISION_MAX_REINTENTOS:
                print(f'[cam]   {VISION_MAX_REINTENTOS} reintentos fallidos. '
                      'El vuelo continúa SIN cámara.')
                break
            if self.caidas == 1:
                # Esto decía "casi siempre es el WiFi del ESP32 bajo carga".
                # El banco del 17-sep lo desmiente: con los hilos de vuelo,
                # MQTT y MiDaS todos corriendo, el caudal NO baja (7.0 fps
                # contra 6.6 del enlace pelado). La carga de la PC no cuesta
                # nada. Lo que sí se midió: abrir la Crazyradio cuesta un
                # 26 % (5.2 fps), y eso es coexistencia en 2.4 GHz.
                print('[cam]   (Si esto se repite: suele ser el GAP8, no la '
                      'distancia ni la carga de la PC —eso está medido en '
                      '`probar_camara.py`—. Mira las NOTAS del final.)')

            # Espera troceada: así el paro de emergencia no tiene que esperar
            # a que termine el backoff.
            t_fin = time.time() + VISION_ESPERA_RECON
            while time.time() < t_fin:
                if self._detener.is_set() or stop_event.is_set():
                    return
                time.sleep(0.1)

    def formato_dominante(self):
        if self.formatos.get(1, 0) > self.formatos.get(0, 0):
            return 'JPEG'
        if self.formatos.get(0, 0) > 0:
            return 'RAW'
        return '?'

    def detener(self):
        self._detener.set()
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()        # cierra su socket con el AI-deck y desbloquea la lectura
            except OSError:
                pass


# ── 2) Estimador de profundidad MiDaS ──
class EstimadorMiDaS:
    ID_DEPTH_ANYTHING = 'depth-anything/Depth-Anything-V2-Small-hf'
    ID_DEPTH_ANYTHING_METRICO = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf'

    def __init__(self, nombre_modelo='MiDaS_small', dispositivo='auto',
                 hilos=VISION_HILOS_TORCH, lado_entrada=None):
        self.lado_entrada = lado_entrada
        # torch ya se importó arriba, en el hilo principal y antes que cv2.
        # Si aquello falló, no seguimos: solo cámara.
        if not _HAY_TORCH or torch is None:
            raise RuntimeError(f'PyTorch no disponible ({_ERROR_TORCH})')

        self.torch = torch
        # Límite de núcleos: evita que la inferencia ahogue al hilo del MoCap
        # y al del teclado (de esto depende que el paro responda al instante).
        # 0 = automático: la mitad de los núcleos lógicos. Más no ayuda
        # (la contienda entre hilos empieza a costar) y deja margen al
        # resto del sistema.
        if not hilos or hilos <= 0:
            hilos = max(1, (os.cpu_count() or 4) // 2)
        try:
            torch.set_num_threads(int(hilos))
            print(f'[midas] torch usando {hilos} hilos')
        except Exception:
            pass

        if dispositivo == 'auto':
            if torch.cuda.is_available():
                dispositivo = 'cuda'
            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                dispositivo = 'mps'
            else:
                dispositivo = 'cpu'
        self.dispositivo = torch.device(dispositivo)
        print(f'[midas] Cargando {nombre_modelo} en {self.dispositivo} ...')

        self.es_metrico = nombre_modelo == 'depth_anything_v2_metrico'
        self.es_depth_anything = nombre_modelo in ('depth_anything_v2',
                                                   'depth_anything_v2_metrico')
        if self.es_depth_anything:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            modelo_id = (self.ID_DEPTH_ANYTHING_METRICO if self.es_metrico
                         else self.ID_DEPTH_ANYTHING)
            try:
                # En vuelo el PC va en el WiFi del AI-deck, sin internet:
                # primero la caché local, sin intentar red (ni esperar timeouts).
                self.transform = AutoImageProcessor.from_pretrained(
                    modelo_id, local_files_only=True)
                self.modelo = AutoModelForDepthEstimation.from_pretrained(
                    modelo_id, local_files_only=True)
            except OSError:
                print('[profundidad] Pesos fuera de la caché; se descargan '
                      '(necesita internet: usa precargar_midas).')
                self.transform = AutoImageProcessor.from_pretrained(modelo_id)
                self.modelo = AutoModelForDepthEstimation.from_pretrained(modelo_id)
            self.modelo.to(self.dispositivo).eval()
            print('[profundidad] Depth Anything V2 Small '
                  + ('MÉTRICO (interiores) ' if self.es_metrico else '') + 'listo.')
            return
        if nombre_modelo == 'midas':
            nombre_modelo = 'MiDaS_small'
        if nombre_modelo not in ('MiDaS_small', 'DPT_Hybrid', 'DPT_Large'):
            raise ValueError(f'Modelo de profundidad desconocido: {nombre_modelo}')
        self.modelo = torch.hub.load('intel-isl/MiDaS', nombre_modelo, trust_repo=True)
        self.modelo.to(self.dispositivo).eval()

        transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
        self.transform = (transforms.small_transform if nombre_modelo == 'MiDaS_small'
                          else transforms.dpt_transform)
        print('[midas] Modelo listo.')

    def inferir(self, img_bgr):
        """
        Mapa float32 del mismo tamaño que la entrada.
        OJO: MiDaS da profundidad INVERSA RELATIVA (disparidad).
        Valor alto = objeto CERCA. No son metros.
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with self.torch.no_grad():
            if self.es_depth_anything:
                if self.lado_entrada is not None:
                    entrada = self.transform(
                        images=img_rgb, return_tensors='pt',
                        size={'height': self.lado_entrada, 'width': self.lado_entrada})
                else:
                    entrada = self.transform(images=img_rgb, return_tensors='pt')
                entrada = {k: v.to(self.dispositivo) for k, v in entrada.items()}
                pred = self.modelo(**entrada).predicted_depth
            else:
                entrada = self.transform(img_rgb).to(self.dispositivo)
                pred = self.modelo(entrada)
            pred = self.torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=img_rgb.shape[:2],
                mode='bicubic', align_corners=False,
            )[0, 0]
        # Modelos relativos: alto = cerca; nunca normalizar por imagen. La
        # escala bruta requiere calibración por modelo.
        salida = pred.cpu().numpy().astype(np.float32, copy=False)
        if self.es_metrico:
            # Este modelo devuelve METROS. Se pasa a 1/m para que siga siendo
            # "alto = cerca" y el detector no note la diferencia: con eso la
            # calibración es a=1, b=0 y no hay que ajustar nada.
            salida = (1.0 / np.maximum(salida, 0.1)).astype(np.float32)
        return salida


def precargar_midas(nombre_modelo=VISION_MODELO):
    """
    Descarga los pesos a ~/.cache/torch para poder usarlos SIN internet.
    Ejecútalo una vez conectado a tu WiFi normal:
        py -c "import sistema_vuelo as p; p.precargar_midas()"
    Para Depth Anything V2 (los pesos van a ~/.cache/huggingface):
        py -c "import sistema_vuelo as p; p.precargar_midas('depth_anything_v2')"
    Después ya puedes conectarte al AP del AI-deck.
    """
    if nombre_modelo == 'hibrido':
        precargar_midas('depth_anything_v2')
        precargar_midas('midas')
        return
    import torch
    ids_hf = {'depth_anything_v2': EstimadorMiDaS.ID_DEPTH_ANYTHING,
              'depth_anything_v2_metrico': EstimadorMiDaS.ID_DEPTH_ANYTHING_METRICO}
    if nombre_modelo in ids_hf:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        print(f'Descargando {ids_hf[nombre_modelo]} a la caché local...')
        AutoImageProcessor.from_pretrained(ids_hf[nombre_modelo])
        AutoModelForDepthEstimation.from_pretrained(ids_hf[nombre_modelo])
        print('Listo. Ya puedes conectarte al WiFi del AI-deck.')
        return
    if nombre_modelo == 'midas':
        nombre_modelo = 'MiDaS_small'
    print(f'Descargando {nombre_modelo} y transforms a la caché local...')
    torch.hub.load('intel-isl/MiDaS', nombre_modelo, trust_repo=True)
    torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
    print('Listo. Ya puedes conectarte al WiFi del AI-deck.')


# ── 3) Utilidades de imagen ──
# Portado tal cual del banco de pruebas: mismas fórmulas, mismos nombres,
# mismas escalas. Lo que ves allí es lo que vuela aquí.

MODOS_COLOR = {
    0: 'color', 1: 'gris', 2: 'canal R', 3: 'canal G', 4: 'canal B',
    5: 'ExG (vegetacion)', 6: 'VARI (vegetacion)', 7: 'falso color',
}
MODOS_WB = {0: 'off', 1: 'gray-world', 2: 'white-patch', 3: 'manual'}

COLORMAPS = [
    ('TURBO', cv2.COLORMAP_TURBO), ('VIRIDIS', cv2.COLORMAP_VIRIDIS),
    ('MAGMA', cv2.COLORMAP_MAGMA), ('INFERNO', cv2.COLORMAP_INFERNO),
    ('JET', cv2.COLORMAP_JET), ('SUMMER', cv2.COLORMAP_SUMMER),
    ('BONE', cv2.COLORMAP_BONE), ('PARULA', cv2.COLORMAP_PARULA),
] if _HAY_CV2 else []

DEMOSAICO = {
    0: ('rapido', PATRONES_BAYER),
    1: ('VNG', {'BG': cv2.COLOR_BayerBG2BGR_VNG, 'GB': cv2.COLOR_BayerGB2BGR_VNG,
                'RG': cv2.COLOR_BayerRG2BGR_VNG, 'GR': cv2.COLOR_BayerGR2BGR_VNG}),
    2: ('EA', {'BG': cv2.COLOR_BayerBG2BGR_EA, 'GB': cv2.COLOR_BayerGB2BGR_EA,
               'RG': cv2.COLOR_BayerRG2BGR_EA, 'GR': cv2.COLOR_BayerGR2BGR_EA}),
} if _HAY_CV2 else {}


ROTACIONES = {0: None, 1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180,
              3: cv2.ROTATE_90_COUNTERCLOCKWISE} if _HAY_CV2 else {}
NOMBRES_ROT = {0: '0', 1: '90', 2: '180', 3: '270'}
NOMBRES_ESPEJO = {0: 'no', 1: 'horizontal', 2: 'vertical'}


def orientar(img, rotacion=0, espejo=0):
    """
    Corrige el montaje físico de la cámara.

    Importa más de lo que parece: MiDaS está entrenado con imágenes
    derechas y usa la posición vertical como pista de distancia. Con la
    escena boca abajo el mapa de profundidad empeora de verdad, no solo
    se ve raro.
    """
    out = img
    r = ROTACIONES.get(rotacion)
    if r is not None:
        out = cv2.rotate(out, r)
    if espejo == 1:
        out = cv2.flip(out, 1)
    elif espejo == 2:
        out = cv2.flip(out, 0)
    return out


def balance_blancos(img, modo=1, gan_r=100, gan_b=100):
    """
    0 off · 1 gray-world · 2 white-patch · 3 manual.

    En vuelo, el MANUAL es el que da resultados comparables entre pasadas:
    los automáticos se reajustan según lo que mire la cámara, así que el
    mismo cultivo cambia de color según el encuadre.
    """
    if modo == 0:
        return img
    res = img.astype(np.float32)
    if modo == 1:
        medias = [res[:, :, c].mean() for c in range(3)]
        objetivo = float(np.mean(medias))
        for c in range(3):
            if medias[c] > 1e-3:
                res[:, :, c] *= objetivo / medias[c]
    elif modo == 2:
        for c in range(3):
            p = float(np.percentile(res[:, :, c], 97))
            if p > 1e-3:
                res[:, :, c] *= 255.0 / p
    elif modo == 3:
        res[:, :, 0] *= gan_b / 100.0
        res[:, :, 2] *= gan_r / 100.0
    return np.clip(res, 0, 255).astype(np.uint8)


def aplicar_saturacion(img, sat=100):
    if sat == 100:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (sat / 100.0), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _normalizar_indice(idx):
    lo, hi = np.percentile(idx, 2), np.percentile(idx, 98)
    if hi - lo < 1e-6:
        return np.zeros(idx.shape, dtype=np.uint8)
    return np.clip((idx - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def indice_exg(img):
    """ExG = 2G - R - B normalizado. Índice de vegetación solo con RGB."""
    f = img.astype(np.float32) + 1e-6
    n = f / f.sum(axis=2, keepdims=True)
    b, g, r = n[:, :, 0], n[:, :, 1], n[:, :, 2]
    return _normalizar_indice(2.0 * g - r - b)


def indice_vari(img):
    """VARI = (G-R)/(G+R-B). Más tolerante a cambios de iluminación."""
    f = img.astype(np.float32)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    den = g + r - b
    den[np.abs(den) < 1e-3] = 1e-3
    return _normalizar_indice((g - r) / den)


def aplicar_modo_color(img, modo, i_colormap=0):
    if modo == 0:
        return img
    if modo == 1:
        return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if modo in (2, 3, 4):
        return cv2.cvtColor(img[:, :, {2: 2, 3: 1, 4: 0}[modo]], cv2.COLOR_GRAY2BGR)
    if modo in (5, 6):
        idx = indice_exg(img) if modo == 5 else indice_vari(img)
        return cv2.applyColorMap(idx, COLORMAPS[i_colormap % len(COLORMAPS)][1])
    if modo == 7:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(g, COLORMAPS[i_colormap % len(COLORMAPS)][1])
    return img


def realzar(img, a):
    """
    Pipeline tonal completo, con el diccionario de ajustes `a`.

    Orden: balance → lineal → realce → saturación → denoise → enfoque.
    El denoise va DESPUÉS de aclarar porque aclarar amplifica el ruido.
    """
    out = img

    if a['wb']:
        out = balance_blancos(out, a['wb'], a['gan_r'], a['gan_b'])

    alpha = a['contraste'] / 100.0
    beta = float(a['brillo'])
    if abs(alpha - 1.0) > 1e-3 or abs(beta) > 0.5:
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    modo = a['realce']
    if modo == 1:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, ca, cb = cv2.split(lab)
        l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
        out = cv2.cvtColor(cv2.merge((l, ca, cb)), cv2.COLOR_LAB2BGR)
    elif modo == 2:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, ca, cb = cv2.split(lab)
        clip = max(0.1, a['clahe_clip'] / 10.0)
        tile = max(1, a['clahe_tile'])
        l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
        out = cv2.cvtColor(cv2.merge((l, ca, cb)), cv2.COLOR_LAB2BGR)
    elif modo == 3:
        # out = in^g  →  g < 1 ACLARA, g > 1 oscurece
        g = max(0.05, a['gamma'] / 100.0)
        tabla = np.array([((i / 255.0) ** g) * 255 for i in range(256)],
                         dtype=np.uint8)
        out = cv2.LUT(out, tabla)

    if a['saturacion'] != 100:
        out = aplicar_saturacion(out, a['saturacion'])

    if a['denoise'] > 0:
        d = int(a['denoise'])
        out = cv2.bilateralFilter(out, 5, d * 12, d * 12)

    if a['enfoque'] > 0:
        amt = a['enfoque'] / 100.0
        borroso = cv2.GaussianBlur(out, (0, 0), 1.2)
        out = cv2.addWeighted(out, 1.0 + amt, borroso, -amt, 0)

    return out


def cargar_ajustes_vision():
    """
    Devuelve los ajustes de imagen a usar.

    Si VISION_AJUSTES_JSON apunta a un archivo del banco de pruebas, sus
    valores pisan a los del diccionario VISION_AJUSTES. Solo se aceptan las
    claves conocidas, así que un JSON viejo o incompleto no rompe nada.
    """
    ajustes = dict(VISION_AJUSTES)
    candidatas = VISION_AJUSTES_JSON
    if isinstance(candidatas, str):
        candidatas = [candidatas]
    if not candidatas:
        return ajustes

    ruta = next((c for c in candidatas if os.path.isfile(c)), None)
    if ruta is None:
        print('[cam] (Sin JSON del banco; uso VISION_AJUSTES del propio archivo)')
        for c in candidatas:
            print(f'[cam]    buscado en: {os.path.abspath(c)}')
        return ajustes

    try:
        with open(ruta, 'r') as f:
            datos = json.load(f)
        cfg = datos.get('cfg', {})
        aceptadas = [k for k in ajustes if k in cfg]
        for k in aceptadas:
            ajustes[k] = cfg[k]
        print(f'[cam] Ajustes cargados del banco: {ruta}')
        print(f'[cam]   {len(aceptadas)} parametros -> realce={ajustes["realce"]} '
              f'clahe {ajustes["clahe_clip"]}/{ajustes["clahe_tile"]} '
              f'brillo {ajustes["brillo"]} contraste {ajustes["contraste"]} '
              f'wb {MODOS_WB[ajustes["wb"]]} color {MODOS_COLOR[ajustes["modo_color"]]}')
    except Exception as e:
        print(f'[cam] No se pudo leer {ruta} ({e}); uso VISION_AJUSTES.')
    return ajustes


def colorear_profundidad(mapa):
    """Normaliza a 0-255 y aplica magma (claro = cerca)."""
    mn, mx = float(mapa.min()), float(mapa.max())
    if mx - mn < 1e-6:
        norm = np.zeros(mapa.shape, dtype=np.uint8)
    else:
        norm = ((mapa - mn) / (mx - mn) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)


def dibujar_texto(img, texto, y, color=(255, 255, 255), x=8, escala=0.5):
    cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala,
                color, 1, cv2.LINE_AA)


def analizar_obstaculos(mapa_profundidad):
    """
    Solo INFORMATIVO por ahora: divide la franja central en 3 columnas y
    compara la disparidad media. La de MENOR valor es la más despejada
    (valor alto = cerca). No manda nada al dron.
    """
    alto, ancho = mapa_profundidad.shape
    franja = mapa_profundidad[int(alto * 0.25): int(alto * 0.75), :]
    tercio = ancho // 3
    izq = float(np.mean(franja[:, :tercio]))
    cen = float(np.mean(franja[:, tercio:2 * tercio]))
    der = float(np.mean(franja[:, 2 * tercio:]))
    sectores = {'izquierda': izq, 'centro': cen, 'derecha': der}
    return {'sectores': sectores, 'mas_libre': min(sectores, key=sectores.get)}


def resumen_caidas():
    """
    ¿ES DISTANCIA O ES CARGA? Este informe lo decide con datos.

    La idea: comparar la distancia media al receptor de los puntos DONDE SE
    CAYÓ el enlace contra la de TODA la trayectoria volada.

      · Si el dron se cae sistemáticamente LEJOS  → sí es alcance/orientación.
      · Si las caídas están repartidas por igual  → NO es distancia; es carga
        del ESP32, alimentación o interferencia.

    También mira el intervalo entre caídas: muy regular = watchdog/brownout;
    irregular = radio.
    """
    if not CAIDAS_WIFI:
        print('\n[cam] Enlace estable: 0 caídas. ✓')
        return

    ref = np.array(VISION_POS_LAPTOP, dtype=float) if VISION_POS_LAPTOP else np.zeros(3)
    etiqueta = 'la laptop' if VISION_POS_LAPTOP else 'el origen'

    print('\n' + '─' * 62)
    print(f'  CAÍDAS DEL ENLACE WIFI: {len(CAIDAS_WIFI)}')
    print('─' * 62)
    print(f'  {"#":>2}  {"t sesion":>9}  {"frames":>7}  {"dist":>6}  '
          f'{"vbat":>6} {"vmin":>6}   posicion (x, y, z)')

    d_caidas, vmins = [], []
    for c in CAIDAS_WIFI:
        p = np.array([c['pose']['x'], c['pose']['y'], c['pose']['z']], dtype=float)
        d = float(np.linalg.norm(p - ref))
        d_caidas.append(d)
        vb, vm = c.get('vbat'), c.get('vbat_min')
        if vm is not None:
            vmins.append(vm)
        s_vb = f'{vb:5.2f}V' if vb is not None else '   -- '
        s_vm = f'{vm:5.2f}V' if vm is not None else '   -- '
        print(f"  {c['n']:>2}  {c['dur_sesion_s']:>8.1f}s  {c['frames_sesion']:>7}  "
              f"{d:>5.2f}m  {s_vb} {s_vm}   ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f})")

    d_caidas = np.array(d_caidas)
    print('─' * 62)

    # ── ¿BROWNOUT? Esto es lo que explica que el SSID desaparezca ──
    # ── ¿Atascos del GAP8 o caídas del enlace? No es lo mismo ──
    n_atascos = sum(1 for c in CAIDAS_WIFI if c.get('tipo') == 'atasco')
    sin_frames = sum(1 for c in CAIDAS_WIFI
                     if c.get('tipo') == 'atasco' and c.get('frames_sesion', 0) == 0)
    n_lentos = sum(1 for c in CAIDAS_WIFI if c.get('lento'))
    if n_lentos:
        print(f'  CAUDAL BAJO: {n_lentos} de esos atascos los cazó el vigilante '
              'de fps,')
        print(f'  no el timeout: el GAP8 seguía mandando, por debajo de '
              f'{VISION_FPS_MINIMO:.1f} fps.')
        print('  Es el fallo del 16-sep, y antes no lo veía nadie: el socket')
        print('  nunca pasaba 4 s mudo, así que ningún watchdog saltaba y desde')
        print('  la PC parecía que la cámara "iba lenta". De referencia, con')
        print('  este dron el banco da 6.6 fps sin radio y 5.2 fps con la radio')
        print('  abierta; por debajo de 2 no hay explicación buena.')
        if n_lentos >= VISION_VIGILANTE_MAX:
            print('  Llegó al tope de reintentos: reconectar no lo cura. Quita y')
            print('  pon la batería del dron antes del siguiente vuelo.')
        print()
    if n_atascos:
        print(f'  TIPO: {n_atascos} atasco(s) del stream, '
              f'{len(CAIDAS_WIFI) - n_atascos} caída(s) de enlace.')
        print()
        print('  Un ATASCO no es una caída: el punto de acceso siguió en pie y')
        print('  el socket TCP siguió abierto; simplemente dejaron de llegar')
        print('  imágenes. Eso descarta que el ESP32 se esté reiniciando —si se')
        print('  reiniciara, la red desaparecería de la lista de WiFi— y apunta')
        print('  al GAP8, que es quien captura y envía.')
        if sin_frames:
            print()
            print(f'  Y {sin_frames} de esos atascos recibieron CERO frames tras')
            print('  reconectar. O sea que el GAP8 ya no reanuda el stream: acepta')
            print('  la conexión (eso lo hace el ESP32) pero no manda nada.')
            print('  Qué hacer, por orden:')
            print('   · Batería cargada. El GAP8 comparte alimentación y es lo')
            print('     que más corriente pide al capturar; con la LiPo baja se')
            print('     cuelga aunque el ESP32 aguante.')
            print('   · Reiniciar el dron entre pruebas para sacar al GAP8 del')
            print('     estado colgado (un reset del ESP32 solo no basta).')
            print('   · JPEG en vez de RAW: menos trabajo por frame para el GAP8.')
            print('   · Actualizar el firmware del GAP8 y del NINA.')
        print()

    if vmins:
        with _vbat_lock:
            todas = [v for (_t, v, _s) in VBAT_LOG]
        vmins = np.array(vmins)
        print(f'  BATERÍA')
        print(f'    Mínimo en las caídas : {vmins.mean():.2f} V de media '
              f'(peor {vmins.min():.2f} V)')
        if todas:
            todas = np.array(todas)
            print(f'    Durante todo el vuelo: media {todas.mean():.2f} V, '
                  f'mínimo {todas.min():.2f} V')

            # Con una o dos caídas no se puede concluir nada: cualquier
            # coincidencia es casualidad. Antes esto imprimía "1/1 coinciden"
            # y acto seguido "la tensión NO se hunde", que se contradecía.
            rango = float(todas.max() - todas.min())
            if len(vmins) < 3:
                print(f'    (Solo {len(vmins)} caída(s): muy pocas para concluir.')
                print('     Repite el vuelo; con 3 o más el veredicto es fiable.)')
            elif rango < 0.05:
                print(f'    (La tensión apenas varió en todo el vuelo, {rango:.2f} V')
                print('     de rango: el dron no llegó a volar de verdad.)')
            else:
                p10 = float(np.percentile(todas, 10))
                n_bajas = int(np.sum(vmins <= p10))
                print(f'    Caídas coincidentes con un hundimiento: '
                      f'{n_bajas}/{len(vmins)}')
                if n_atascos >= len(CAIDAS_WIFI) * 0.6:
                    print('    → OJO: casi todo fueron ATASCOS, no caídas de enlace,')
                    print('      así que el ESP32 NO se estaba reiniciando y esto no')
                    print('      es un brownout del WiFi. Aun así la tensión está')
                    print(f'      en {vmins.mean():.2f} V, que es demasiado baja para')
                    print('      volar: cambia la batería y vuelve a medir.')
                elif vmins.mean() < 3.4 or n_bajas >= len(vmins) * 0.6:
                    print('    → BROWNOUT casi seguro. El ESP32 se resetea por falta')
                    print('      de tensión y por eso desaparece el SSID. Acciones:')
                    print('        · batería recién cargada y en buen estado,')
                    print('        · LED_BRILLO más bajo o LED_ACTIVO = False,')
                    print('        · quitar peso/decks que no uses en esta prueba,')
                    print('        · JPEG en vez de RAW (menos corriente de TX).')
                else:
                    print('    → La tensión NO se hunde en las caídas. Descarta')
                    print('      brownout por batería: mira el conector del deck')
                    print('      (vibración) y el firmware del ESP32.')
        print()

    # Comparación con toda la trayectoria volada
    try:
        traj = np.asarray(real_trajectory, dtype=float)
        if len(traj) > 10:
            d_traj = np.linalg.norm(traj - ref, axis=1)
            m_caida, m_traj = float(d_caidas.mean()), float(d_traj.mean())
            print(f'  Distancia media a {etiqueta}:')
            print(f'    en las caídas     : {m_caida:.2f} m')
            print(f'    en todo el vuelo  : {m_traj:.2f} m  (máx {d_traj.max():.2f} m)')
            print()
            # Percentil: ¿en qué parte del rango volado cayeron?
            pct = float(np.mean(d_traj < m_caida) * 100)
            print(f'    Las caídas ocurrieron más lejos que el {pct:.0f}% del vuelo.')
            if pct >= 75 and m_caida > m_traj * 1.25:
                print('    → SÍ apunta a DISTANCIA / orientación de la antena.')
                print('      Prueba: acerca el receptor, o pasa el deck a modo')
                print('      estación con el router más cerca del área de vuelo.')
            elif pct <= 60:
                print('    → NO es distancia: se cae por igual cerca y lejos.')
                print('      Mira carga (RAW vs JPEG), alimentación e interferencia.')
            else:
                print('    → No concluyente. Repite el vuelo para tener más caídas.')
        else:
            print('  (Sin trayectoria suficiente para comparar.)')
    except Exception as e:
        print(f'  (No se pudo comparar con la trayectoria: {e})')

    # Regularidad temporal
    if len(CAIDAS_WIFI) >= 3 and n_atascos < len(CAIDAS_WIFI) * 0.5:
        # Solo tiene sentido con caídas reales: las sesiones que corta el
        # watchdog duran siempre VISION_TIMEOUT_LECTURA y falsearían el dato.
        dur = np.array([c['dur_sesion_s'] for c in CAIDAS_WIFI[1:]])
        if dur.mean() > 0:
            cv = float(dur.std() / dur.mean())
            print()
            print(f'  Duración de cada sesión: media {dur.mean():.1f} s, '
                  f'variabilidad {cv:.2f}')
            if cv < 0.35:
                print('    → Muy regular. Huele a watchdog o brownout del ESP32,')
                print('      no a radio. Revisa batería y alimentación.')
            else:
                print('    → Irregular. Típico de radio: interferencia u obstrucción.')

    print('─' * 62)
    try:
        os.makedirs(VISION_DIR_CAPTURAS, exist_ok=True)
        ruta = os.path.join(VISION_DIR_CAPTURAS, 'caidas_wifi.json')
        with open(ruta, 'w') as f:
            json.dump(CAIDAS_WIFI, f, indent=2)
        print(f'  Detalle guardado en: {ruta}')
    except Exception:
        pass
    print('─' * 62)


# ── 4) Sistema de visión completo ──
class SistemaVision:
    """
    Arranca/para los tres hilos y expone el último resultado.
    Uso:
        vision = SistemaVision(); vision.iniciar()
        ...  vuelo normal ...
        vision.detener()
    """

    def __init__(self):
        # Ajustes de imagen: del JSON del banco si existe, si no del
        # diccionario VISION_AJUSTES. Se leen UNA vez, al construir.
        self.ajustes = cargar_ajustes_vision()
        self.cola = queue.Queue(maxsize=1)
        self.receptor = None
        self.hilo_proc = None
        self.hilo_midas = None
        self.hilo_ventana = None
        self._detener = threading.Event()
        self._lock = threading.Lock()
        self._ultimo = {'img': None, 'img_base': None, 'mapa': None, 'mapa_dist': None, 'info': None, 'ts': 0.0}
        self._lock_midas = threading.Lock()
        self._entrada_midas = None     # ranura de 1: la imagen más reciente
        self._mapa_ts = 0.0            # captura monotónica, NO llegada del vídeo
        self._mapa = None
        self._mapa_dist = None
        self._info = None
        self.estimador = None
        self.estimador_dist = None
        self.midas_listo = False
        self.fps_cam = 0.0
        self.fps_midas = 0.0
        self.ms_det = 0.0
        self.ms_dist = 0.0
        self.ventana_abierta = False
        # Petición de ventana. Se puede poner desde cualquier hilo; quien la
        # crea es SIEMPRE el hilo de la ventana (HighGUI de Windows no tolera
        # que otro hilo toque sus ventanas). Arranca pedida.
        self._quiere_ventana = threading.Event()
        self._quiere_ventana.set()
        self.n_capturas = 0
        self._t_ultima_captura = 0.0
        self._t_informe = 0.0

    # -- API ------------------------------------------------------------
    def iniciar(self):
        if not VISION_ACTIVA:
            return False
        if not _HAY_CV2:
            print('⚠ Visión desactivada: falta OpenCV (pip install opencv-python).')
            return False
        if VISION_MIDAS and not _HAY_TORCH:
            _explicar_error_torch(_ERROR_TORCH)
        try:
            self.receptor = HiloReceptorAIDeck(
                VISION_IP, VISION_PUERTO, self.cola,
                bayer=not VISION_MONO, patron=self.ajustes['patron'],
                demosaico=self.ajustes['demosaico'])
            self.receptor.start()

            self.hilo_proc = threading.Thread(target=self._bucle_procesado,
                                              daemon=True, name='procesado')
            self.hilo_proc.start()

            self.hilo_midas = threading.Thread(target=self._bucle_midas,
                                               daemon=True, name='midas')
            self.hilo_midas.start()

            self.hilo_ventana = threading.Thread(target=self._bucle_ventana,
                                                 daemon=True, name='ventana-cam')
            self.hilo_ventana.start()
            # Registro global: lo usa el hilo del teclado de la consola para
            # reabrir la ventana con 'v'. Sin esto, cerrar la ventana con la X
            # dejaba la cámara sin ninguna forma de volver a verse.
            global VISION_EN_CURSO
            VISION_EN_CURSO = self
            print('✓ Visión AI-deck iniciada '
                  "(en la ventana: 'q'/ESC = PARO DE EMERGENCIA, 's' = captura)")
            return True
        except Exception as e:
            print(f'⚠ No se pudo iniciar la visión: {e} — el vuelo continúa.')
            return False

    def detener(self):
        self._detener.set()
        try:
            if self.receptor:
                self.receptor.detener()
        except Exception:
            pass
        for h in (self.hilo_proc, self.hilo_midas, self.hilo_ventana):
            try:
                if h:
                    h.join(timeout=2.0)
            except Exception:
                pass
        try:
            if self.receptor:
                self.receptor.join(timeout=2.0)
        except Exception:
            pass
        try:
            resumen_caidas()     # informe: ¿distancia o carga?
        except Exception as e:
            print(f'[cam] No se pudo generar el resumen de caídas: {e}')

    def ultimo(self):
        """Copia superficial del último resultado (thread-safe)."""
        with self._lock:
            return dict(self._ultimo)

    # -- hilo de procesado (el ÚNICO que llama a MiDaS) -----------------
    def _bucle_procesado(self):
        """
        Consume frames y publica la imagen. NO llama a MiDaS.

        Esto era el cuello de botella de los fps: antes, este mismo hilo
        hacía la inferencia. La cola es de tamaño 1 y este es su único
        consumidor, así que mientras MiDaS trabajaba (1-3 s en esta CPU) no
        se sacaba ningún frame nuevo. Resultado: el vídeo iba al ritmo de
        MiDaS —menos de 1 fps— y no al de la cámara.

        Ahora este bucle solo orienta, realza y publica: se queda a la
        velocidad del enlace. MiDaS vive en `_bucle_midas` y va a su aire.
        """
        t_prev = time.time()

        while not self._detener.is_set() and not stop_event.is_set():
            try:
                fr = self.cola.get(timeout=0.2)
            except queue.Empty:
                continue
            if fr is None:
                continue

            try:
                # Orientación primero: todo lo que sigue (realce, MiDaS,
                # análisis de obstáculos, capturas) ve la escena derecha.
                img_base = orientar(fr.imagen, self.ajustes['rotacion'],
                                    self.ajustes['espejo'])
                # Pipeline tonal (balance, brillo/contraste, CLAHE/gamma,
                # saturación, denoise, enfoque) con los ajustes del banco.
                img_tonal = realzar(img_base, self.ajustes)

                # El modo de color es SOLO para ver. A MiDaS se le pasa la
                # imagen tonal: un índice de vegetación o un falso color lo
                # confundirían, porque no es una foto de la escena.
                if self.ajustes['modo_color'] != 0:
                    img = aplicar_modo_color(img_tonal, self.ajustes['modo_color'],
                                             self.ajustes['colormap'])
                else:
                    img = img_tonal

                ahora = time.time()
                dt = ahora - t_prev
                t_prev = ahora
                if dt > 0:
                    self.fps_cam = 0.9 * self.fps_cam + 0.1 * (1.0 / dt)

                # Informe periódico en consola: los fps son un número, no
                # una impresión. Si dice 6 fps, no hay nada que optimizar en
                # la PC: es el techo del enlace con RAW.
                if ahora - self._t_informe >= 5.0:
                    self._t_informe = ahora
                    rx = self.receptor.frames_recibidos if self.receptor else 0
                    kbs = 0.0
                    if self.receptor and self.receptor.t_inicio:
                        dur = ahora - self.receptor.t_inicio
                        kbs = (self.receptor.bytes_totales / 1024.0 / dur) if dur > 0 else 0.0
                    fmt = self.receptor.formato_dominante() if self.receptor else '?'
                    extra = ''
                    if fmt == 'RAW':
                        extra = '  (RAW satura el enlace: pasa a JPEG en el GAP8)'
                    prof = f' | prof {self.fps_midas:3.1f}/s (det {self.ms_det:.0f} ms'
                    if self.estimador_dist is not None:
                        prof += f', dist {self.ms_dist:.0f} ms'
                    prof += ')'
                    print(f'[cam] {self.fps_cam:4.1f} fps | {kbs:4.0f} kB/s | '
                          f'{fmt} | frames {rx}{prof}{extra}')

                # Publicar el vídeo YA, con el último mapa disponible.
                with self._lock:
                    self._ultimo = {'img': img, 'img_base': img_base, 'mapa': self._mapa,
                                    'mapa_dist': self._mapa_dist,
                                    'info': self._info, 'ts': ahora,
                                    'mapa_ts': self._mapa_ts}

                # Dejar la imagen tonal para MiDaS: una sola ranura, siempre
                # la más reciente. Si MiDaS va lento, se pierden frames
                # intermedios, que es justo lo que queremos.
                with self._lock_midas:
                    self._entrada_midas = (img_tonal, fr.monotonic_ts)

                if VISION_GUARDAR_CADA > 0 and ahora - self._t_ultima_captura >= VISION_GUARDAR_CADA:
                    self._t_ultima_captura = ahora
                    self._guardar(img, self._mapa, img_base=img_base)
            except Exception as e:
                # Un fallo de visión NUNCA debe tumbar el vuelo.
                print(f'[cam] Error procesando frame: {e}')
                time.sleep(0.1)

    # -- hilo de MiDaS (aparte, para no frenar el vídeo) ------------------
    def _bucle_midas(self):
        _bajar_prioridad_hilo()

        # El modelo se carga AQUÍ: aunque tarde 20 s, ni el vuelo ni el
        # vídeo esperan.
        if VISION_MIDAS:
            if not _HAY_TORCH:
                print('[midas] Sin PyTorch → solo cámara, sin profundidad.')
                return
            try:
                if VISION_MODELO == 'hibrido':
                    self.estimador = EstimadorMiDaS('depth_anything_v2', VISION_DISPOSITIVO,
                                                  lado_entrada=VISION_DA_LADO)
                    self.estimador_dist = EstimadorMiDaS('midas', VISION_DISPOSITIVO)
                else:
                    self.estimador = EstimadorMiDaS(VISION_MODELO, VISION_DISPOSITIVO,
                                                  lado_entrada=VISION_DA_LADO)
                self.midas_listo = True
            except Exception as e:
                self.estimador = None
                print(f'[midas] ⚠ No se pudo cargar el modelo: {e}')
                print('[midas]   Se seguirá mostrando solo la cámara.')
                return
        else:
            return

        t_ultima_inf = 0.0
        while not self._detener.is_set() and not stop_event.is_set():
            ahora = time.time()
            if ahora - t_ultima_inf < VISION_MIN_PERIODO:
                time.sleep(0.02)
                continue
            with self._lock_midas:
                img = self._entrada_midas
                self._entrada_midas = None
            if img is None:
                time.sleep(0.02)
                continue
            img, captura_ts = img
            try:
                t0 = time.time()
                t_det = time.perf_counter()
                mapa = self.estimador.inferir(img)
                ms = (time.perf_counter() - t_det) * 1000.0
                self.ms_det = 0.8 * self.ms_det + 0.2 * ms
                mapa_dist = None
                if self.estimador_dist is not None:
                    t_dist = time.perf_counter()
                    mapa_dist = self.estimador_dist.inferir(img)
                    ms = (time.perf_counter() - t_dist) * 1000.0
                    self.ms_dist = 0.8 * self.ms_dist + 0.2 * ms
                dt_inf = time.time() - t0
                self.fps_midas = 1.0 / dt_inf if dt_inf > 0 else 0.0
                t_ultima_inf = time.time()
                info = analizar_obstaculos(mapa)
                with self._lock:
                    self._mapa, self._info = mapa, info
                    self._mapa_dist = mapa_dist
                    self._mapa_ts = captura_ts
                    self._ultimo['mapa'] = mapa
                    self._ultimo['mapa_dist'] = mapa_dist
                    self._ultimo['info'] = info
                    self._ultimo['mapa_ts'] = captura_ts
            except Exception as e:
                print(f'[midas] error: {e}')
                time.sleep(0.2)

    # -- hilo de ventana (NO hace inferencia: por eso la tecla va fluida) --
    TITULO_VENTANA = 'AI-deck  |  camara / profundidad MiDaS   [q o ESC = PARO]'

    def mostrar_ventana(self):
        """
        Pide que la ventana se abra (o se vuelva a abrir). Segura desde
        cualquier hilo: solo levanta una bandera, y el hilo de la ventana
        hace el trabajo.
        """
        self._quiere_ventana.set()

    def _crear_ventana(self, titulo):
        """Crea la ventana de cero, barriendo antes cualquier resto."""
        try:
            cv2.destroyWindow(titulo)
        except Exception:
            pass
        cv2.namedWindow(titulo, cv2.WINDOW_AUTOSIZE)
        cv2.waitKey(1)

    def _ventana_viva(self, titulo):
        try:
            return cv2.getWindowProperty(titulo, cv2.WND_PROP_VISIBLE) >= 1
        except Exception:
            return False

    def _bucle_ventana(self):
        # Antes este hilo abría la ventana UNA vez y hacía `break` al ver que
        # la habías cerrado: el hilo moría y la ventana no volvía nunca, ni
        # aunque la pidieras. Ahora cerrar la ventana solo la duerme.
        titulo = self.TITULO_VENTANA
        abierta = False
        aviso_cierre = False
        panel_cache = None
        ts_cache = -1.0
        t_refresco = 0.0
        while not self._detener.is_set() and not stop_event.is_set():
            # ── ¿hay que abrir (o reabrir) la ventana? ──
            if not abierta:
                if not self._quiere_ventana.is_set():
                    time.sleep(0.2)      # dormida: no gasta nada
                    continue
                try:
                    self._crear_ventana(titulo)
                except Exception as e:
                    print(f'[cam] ⚠ No se pudo abrir la ventana: {e}')
                    self._quiere_ventana.clear()
                    self.ventana_abierta = False
                    time.sleep(1.0)
                    continue
                abierta = True
                aviso_cierre = False
                panel_cache = None       # el panel viejo era de otra ventana
                ts_cache = -1.0
                self.ventana_abierta = True

            try:
                # El cierre se comprueba AQUÍ, antes de nada. Antes se
                # comprobaba al final del ciclo, y la rama de caché salía por
                # `continue` sin llegar: con la escena quieta, la ventana
                # cerrada no se detectaba.
                if not self._ventana_viva(titulo):
                    abierta = False
                    self.ventana_abierta = False
                    self._quiere_ventana.clear()
                    panel_cache = None
                    try:
                        cv2.destroyWindow(titulo)
                        cv2.waitKey(1)
                    except Exception:
                        pass
                    if not aviso_cierre:
                        print('⚠ Ventana de la camara cerrada. El vuelo y la '
                              'visión siguen.')
                        print("   PARO DE EMERGENCIA: 'q' EN LA CONSOLA.   "
                              "Reabrir la ventana: 'v' EN LA CONSOLA.")
                        aviso_cierre = True
                    continue

                u = self.ultimo()
                img, mapa, info = u['img'], u['mapa'], u['info']

                # Recomponer SOLO si hay frame nuevo (o cada 0.5 s para
                # refrescar contadores). Antes se rehacía el panel entero
                # ~33 veces por segundo aunque la imagen fuera la misma:
                # hstack + resize + colormap + textos, para nada. Esa CPU
                # se la quitaba al hilo que vacía el socket de la cámara.
                ahora_v = time.time()
                if (panel_cache is not None and u['ts'] == ts_cache
                        and ahora_v - t_refresco < 0.5):
                    cv2.imshow(titulo, panel_cache)
                    k = cv2.waitKey(30) & 0xFF     # el paro sigue igual de vivo
                    if k in (ord('q'), ord('Q'), 27):
                        activar_paro_emergencia('ventana de la camara')
                        break
                    if k in (ord('s'), ord('S')):
                        self._guardar(img, mapa, img_base=u.get('img_base'))
                    continue
                ts_cache = u['ts']
                t_refresco = ahora_v

                if img is None:
                    # Sin señal todavía: pintamos un panel negro. Lo importante
                    # es que la ventana siga viva para que 'q' responda.
                    panel = np.zeros((ALTO_CAM, ANCHO_CAM * 2, 3), dtype=np.uint8)
                    if self.receptor and self.receptor.conectado:
                        estado = 'esperando imagenes...'
                    elif self.receptor and self.receptor.caidas:
                        estado = f'RECONECTANDO... (caidas: {self.receptor.caidas})'
                    else:
                        estado = 'SIN CONEXION con el AI-deck'
                    dibujar_texto(panel, estado, 30, (0, 200, 255))
                    dibujar_texto(panel, "'q' o ESC = PARO DE EMERGENCIA", 55)
                else:
                    if mapa is not None:
                        panel = np.hstack([img, colorear_profundidad(mapa)])
                    else:
                        panel = img.copy()

                if VISION_ESCALA != 1:
                    panel = cv2.resize(panel, None, fx=VISION_ESCALA, fy=VISION_ESCALA,
                                       interpolation=cv2.INTER_NEAREST)

                rx = self.receptor.frames_recibidos if self.receptor else 0
                cai = self.receptor.caidas if self.receptor else 0
                ata = self.receptor.atascos if self.receptor else 0
                dibujar_texto(panel, f'cam {self.fps_cam:4.1f} fps | rx {rx} | '
                                     f'midas {self.fps_midas:4.1f} fps | '
                                     f'caidas {cai} atascos {ata}', 20,
                              (255, 255, 255) if cai + ata == 0 else (0, 200, 255))

                # Aviso de congelamiento en pantalla: si el último frame es
                # viejo, decirlo. Antes había que adivinar si la imagen estaba
                # quieta porque la escena no se movía o porque no llegaba nada.
                tuf = self.receptor.t_ultimo_frame if self.receptor else None
                if tuf is not None and time.time() - tuf > 1.5:
                    dibujar_texto(panel,
                                  f'IMAGEN CONGELADA: {time.time()-tuf:.1f} s sin frames',
                                  panel.shape[0] // 2, (0, 80, 255),
                                  x=max(8, panel.shape[1] // 8), escala=0.8)
                if info is not None:
                    s = info['sectores']
                    dibujar_texto(panel,
                                  f"I:{s['izquierda']:.0f}  C:{s['centro']:.0f}  "
                                  f"D:{s['derecha']:.0f}  -> libre: {info['mas_libre']}", 40)
                dibujar_texto(panel, f"x{mocap_pose['x']:+.2f} y{mocap_pose['y']:+.2f} "
                                     f"z{mocap_pose['z']:+.2f}", 60, (180, 255, 180))
                a = self.ajustes
                dibujar_texto(panel,
                              f"realce {a['realce']} clahe {a['clahe_clip']}/{a['clahe_tile']} "
                              f"br {a['brillo']:+d} ct {a['contraste']} | "
                              f"wb {MODOS_WB[a['wb']]} | {MODOS_COLOR[a['modo_color']]} | "
                              f"rot {NOMBRES_ROT[a['rotacion']]} esp {NOMBRES_ESPEJO[a['espejo']]}",
                              78, (160, 255, 255))
                dibujar_texto(panel, "q / ESC = PARO DE EMERGENCIA    s = captura",
                              panel.shape[0] - 12, (80, 80, 255))

                panel_cache = panel
                cv2.imshow(titulo, panel)

                # ── AQUÍ está la clave del paro de emergencia ──
                # Este waitKey se ejecuta ~30 veces por segundo pase lo que
                # pase, porque este hilo no hace nada pesado.
                tecla = cv2.waitKey(30) & 0xFF
                if tecla in (ord('q'), ord('Q'), 27):        # 27 = ESC
                    activar_paro_emergencia('ventana de la camara')
                    break
                if tecla in (ord('s'), ord('S')):
                    self._guardar(img, mapa, img_base=u.get('img_base'))
                # El cierre con la X se detecta al principio del ciclo
                # siguiente, que es el único sitio por el que pasan TODAS las
                # ramas.
            except Exception as e:
                print(f'[cam] Error en la ventana: {e}')
                time.sleep(0.2)

        self.ventana_abierta = False
        try:
            cv2.destroyWindow(titulo)
            cv2.waitKey(1)
        except Exception:
            pass

    # -- capturas --------------------------------------------------------
    def _guardar(self, img, mapa, img_base=None):
        if img is None:
            return
        try:
            os.makedirs(VISION_DIR_CAPTURAS, exist_ok=True)
            self.n_capturas += 1
            n = self.n_capturas
            base = os.path.join(VISION_DIR_CAPTURAS, f'{n:04d}')
            cv2.imwrite(f'{base}_cam.png', img)
            if img_base is not None:
                cv2.imwrite(f'{base}_crudo.png', img_base)
            if mapa is not None:
                cv2.imwrite(f'{base}_prof.png', colorear_profundidad(mapa))
                np.save(f'{base}_prof.npy', mapa)
            ajustes = dict(self.ajustes)
            for clave, valor in ajustes.items():
                try:
                    json.dumps(valor)
                except (TypeError, ValueError, OverflowError):
                    ajustes[clave] = str(valor)
            with open(f'{base}_pose.json', 'w') as f:
                json.dump({'t': time.time(), 'pose': dict(mocap_pose),
                           'modelo': VISION_MODELO, 'ajustes': ajustes}, f)
            print(f'[cam] Captura {n:04d} guardada')
        except Exception as e:
            print(f'[cam] Error guardando captura: {e}')


# ─────────────────────────────────────────────
# TABLA DE CULTIVOS
# ─────────────────────────────────────────────
CULTIVOS = {
    '1': {'nombre': 'Cardamomo',       'altura': 0.3, 'land_dur': 2.0},
    '2': {'nombre': 'Cafe',            'altura': 0.5, 'land_dur': 3.0},
    '3': {'nombre': 'Cana de azucar',  'altura': 0.8, 'land_dur': 4.0},
    '4': {'nombre': 'Banano',          'altura': 0.8, 'land_dur': 4.0},
    '5': {'nombre': 'Palma de aceite', 'altura': 1.0, 'land_dur': 5.0},
}

def seleccionar_cultivo_tk():
    """
    Ventana tkinter para seleccionar el cultivo a regar.
    Retorna dict con nombre, altura y land_dur, o None si canceló.
    """
    import tkinter as tk

    resultado = [None]

    root = tk.Tk()
    root.title('Seleccionar cultivo')
    root.configure(bg='#0d0d1a')
    root.attributes('-topmost', True)
    root.resizable(False, False)

    tk.Label(root, text='CULTIVO A REGAR',
             bg='#0d0d1a', fg='#00aaff', font=('Arial', 13, 'bold')).pack(pady=(16, 4), padx=30)
    tk.Label(root, text='Selecciona el cultivo para ajustar la altura de vuelo:',
             bg='#0d0d1a', fg='#888899', font=('Arial', 9)).pack(pady=(0, 10))

    frame = tk.Frame(root, bg='#0d0d1a')
    frame.pack(padx=24, pady=4)

    # Encabezado tabla
    tk.Label(frame, text='Cultivo', bg='#131D35', fg='#00BFA5',
             font=('Arial', 9, 'bold'), width=18, anchor='w',
             padx=8).grid(row=0, column=0, sticky='ew', padx=1, pady=1)
    tk.Label(frame, text='Altura (m)', bg='#131D35', fg='#00BFA5',
             font=('Arial', 9, 'bold'), width=10, anchor='center').grid(row=0, column=1, sticky='ew', padx=1, pady=1)
    tk.Label(frame, text='Aterrizaje', bg='#131D35', fg='#00BFA5',
             font=('Arial', 9, 'bold'), width=10, anchor='center').grid(row=0, column=2, sticky='ew', padx=1, pady=1)

    for i, (k, c) in enumerate(CULTIVOS.items()):
        bg = '#0D1525' if i % 2 == 0 else '#131D35'
        tk.Label(frame, text=c['nombre'], bg=bg, fg='#E8F0FE',
                 font=('Arial', 10), width=18, anchor='w',
                 padx=8).grid(row=i+1, column=0, sticky='ew', padx=1, pady=1)
        tk.Label(frame, text=f"{c['altura']:.1f} m", bg=bg, fg='white',
                 font=('Arial', 10, 'bold'), width=10,
                 anchor='center').grid(row=i+1, column=1, sticky='ew', padx=1, pady=1)
        tk.Label(frame, text=f"{c['land_dur']:.0f} s", bg=bg, fg='#9AA5B4',
                 font=('Arial', 9), width=10,
                 anchor='center').grid(row=i+1, column=2, sticky='ew', padx=1, pady=1)

    tk.Label(root, text='', bg='#0d0d1a').pack(pady=4)

    btn_frame = tk.Frame(root, bg='#0d0d1a')
    btn_frame.pack(padx=24, pady=4)

    for k, c in CULTIVOS.items():
        def _seleccionar(cultivo=c):
            resultado[0] = cultivo
            root.destroy()
        tk.Button(btn_frame, text=f"  {k}. {c['nombre']}  ({c['altura']} m)",
                  command=_seleccionar,
                  bg='#003311', fg='#00ff88', font=('Arial', 10),
                  relief='flat', padx=10, pady=6, width=26,
                  anchor='w').pack(pady=3)

    tk.Button(root, text='Cancelar', command=root.destroy,
              bg='#1a0000', fg='#ff4444', font=('Arial', 9),
              relief='flat', padx=10, pady=5).pack(pady=(6, 14))

    root.mainloop()
    return resultado[0]


def ejecutar_ruta_pursuit(scf, ruta_suave_2d, cultivo):
    """
    Vuela la ruta con Pure Pursuit: el lazo de posición se cierra AQUÍ, en
    el PC, contra el MoCap, y al dron se le mandan velocidades de mundo a
    PP_CONTROL_HZ.

    No toca `ejecutar_ruta_fisica` ni `ejecutar_ruta_trayectoria`: si algo
    falla devuelve False y el vuelo cae al método de siempre.

    Ley de control, por ciclo:
        v_xy = rapidez * T(s_proj + L)  +  KP_LAT * e_lat
        v_z  = rapidez * T_z            +  KP_Z * (z_ref - z)
    """
    try:
        import pursuit_ruta as pp_mod
    except Exception as e:
        print(f'⚠ No se pudo importar pursuit_ruta.py: {e}')
        return False

    altura   = cultivo['altura']
    land_dur = cultivo['land_dur']

    # ── Ruta 3D a altura constante, remuestreada fina ──
    ruta = np.asarray(ruta_suave_2d, dtype=float)
    path = np.column_stack([ruta[:, 0], ruta[:, 1],
                            np.full(len(ruta), float(altura))])
    path = pp_mod.remuestrear(path, PP_PASO_RUTA)

    pp = pp_mod.PurePursuit(L=PP_LOOKAHEAD_T * PP_V_MAX,
                            L_min=PP_L_MIN, L_max=PP_L_MAX,
                            T_look=PP_LOOKAHEAD_T,
                            tol_meta=PP_TOL_META).set_path(path)

    v_prof = pp_mod.perfil_velocidad(pp.s, pp.kappa, pp.T,
                                     v_max=PP_V_MAX, v_min=PP_V_MIN,
                                     a_lat=PP_A_LAT_MAX, a_tan=PP_A_TAN_MAX,
                                     vz_max=PP_VZ_MAX)
    t_est = float(np.sum(np.diff(pp.s) / np.maximum(v_prof[:-1], 1e-3)))
    k_max = float(pp.kappa.max())

    print()
    print('─' * 62)
    print('  PURE PURSUIT  (lazo cerrado en el PC contra el MoCap)')
    print('─' * 62)
    print(f'  Puntos de ruta   : {len(pp.path)}  ({pp.length:.2f} m, '
          f'paso {pp.length/max(len(pp.path)-1,1)*100:.1f} cm)')
    print(f'  Radio mínimo     : {1.0/max(k_max,1e-6):.2f} m  (kappa {k_max:.2f} 1/m)')
    print(f'  Velocidad        : {v_prof.min():.2f} – {v_prof.max():.2f} m/s '
          f'(media {v_prof.mean():.2f})')
    print(f'  Tiempo estimado  : {t_est:.1f} s')
    print(f'  L = {PP_LOOKAHEAD_T:.2f}s·v ∈ [{PP_L_MIN:.2f}, {PP_L_MAX:.2f}] m | '
          f'KP_LAT={PP_KP_LAT} KP_Z={PP_KP_Z} | {PP_CONTROL_HZ:.0f} Hz | '
          f'yaw {PP_YAW_MODE}')
    print('─' * 62)

    # ── Esperar posición del MoCap ──
    print('Esperando posicion inicial del MoCap...')
    t0 = time.time()
    while (mocap_pose['x'] == 0.0 and mocap_pose['y'] == 0.0
           and mocap_pose['z'] == 0.0):
        if stop_event.is_set():
            return True
        if time.time() - t0 > 10.0:
            print('Timeout esperando MoCap.')
            return False
        time.sleep(0.05)

    cmd = scf.cf.commander

    def enviar(vx, vy, vz, yr=0.0):
        """
        CARRERA CRÍTICA, y por poco se me escapa: el paro de emergencia
        llama a send_stop_setpoint() desde OTRO hilo (la consola o la
        ventana de la cámara). Si este lazo manda un setpoint de velocidad
        justo después, el dron REARMA los motores —un setpoint de velocidad
        cero no es "motores apagados", es "quédate flotando aquí"— y el
        paro de emergencia no para nada.
        Por eso se comprueba stop_event ANTES de cada envío: en cuanto está
        puesto, esta función deja de tocar la radio para siempre.
        """
        if stop_event.is_set():
            return False
        cmd.send_velocity_world_setpoint(
            float(vx), float(vy), float(vz), float(yr))
        return True

    def frenar(n=5):
        """
        Velocidad cero repetida (un paquete suelto se puede perder).
        NO se usa para el paro de emergencia: ahí lo que corta es
        send_stop_setpoint(), y esto no debe rearmar nada.
        """
        if stop_event.is_set():
            return
        for _ in range(n):
            if not enviar(0.0, 0.0, 0.0, 0.0):
                return
            time.sleep(0.02)

    # ── Despegue, aproximación, hover y descenso ──
    # Portados TAL CUAL de flight_purepursuit.py. Los había reescrito a mi
    # manera y era un error: son justo las partes que llevan 15 vuelos de
    # ajuste encima, y las diferencias que yo había metido no eran mejoras,
    # eran regresiones. Ver las notas de cada una.
    periodo = 1.0 / PP_CONTROL_HZ

    def despegar(z_obj, vz=0.10, tol=0.05, timeout=10.0):
        """
        Sube en VERTICAL PURO: vx = vy = 0, nada de corrección horizontal.
        Esto es deliberado en el original y yo lo había roto: mi versión
        corregía en XY desde el primer instante, con el dron todavía
        apoyado en el suelo. Con cualquier desfase del MoCap eso es un
        comando de deslizamiento lateral contra el suelo, o sea volcar.
        Mientras haya contacto, lo único seguro es subir.
        """
        print(f'Despegando (low-level) hasta z={z_obj:.2f} m...')
        t0 = time.time()
        while not stop_event.is_set():
            if mocap_pose['z'] >= z_obj - tol:
                break
            if time.time() - t0 > timeout:
                print('  ⚠ Timeout en despegue — continuando.')
                break
            enviar(0.0, 0.0, vz)
            time.sleep(0.02)
        enviar(0.0, 0.0, 0.0)
        time.sleep(0.1)
        print(f'  Altura alcanzada: z={mocap_pose["z"]:.3f} m')

    def ir_al_inicio(dst, max_v=0.20, kp=0.6, tol=0.04, timeout=15.0):
        """
        Ganancias separadas por eje: kp=0.6 en horizontal y kp_z=0.8
        limitado a ±0.15 m/s en vertical. Yo había puesto 1.5 para los
        tres ejes con un solo límite de 0.25, que es 2.5 veces más agresivo
        en horizontal y casi el doble de velocidad vertical. Más ganancia
        aquí no llega antes: llega oscilando, y entra a la ruta oscilando.
        """
        print(f'Moviéndose al inicio: ({dst[0]:.2f}, {dst[1]:.2f}, {dst[2]:.2f})...')
        t0, kp_z = time.time(), 0.8
        while not stop_event.is_set():
            ex = dst[0] - mocap_pose['x']
            ey = dst[1] - mocap_pose['y']
            ez = dst[2] - mocap_pose['z']
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
            enviar(vx, vy, float(np.clip(kp_z * ez, -0.15, 0.15)))
            time.sleep(0.02)
        enviar(0.0, 0.0, 0.0)
        time.sleep(0.1)
        print(f'  Posición alcanzada: ({mocap_pose["x"]:.3f}, '
              f'{mocap_pose["y"]:.3f}, {mocap_pose["z"]:.3f})')

    def mantener(objetivo, duracion, kp=1.5, kp_z=2.0, v_max=0.35):
        """
        Hover ANCLADO al punto. Mandar velocidad cero no mantiene la
        posición: deja al dron a la deriva.
        """
        t0 = time.time()
        while time.time() - t0 < duracion and not stop_event.is_set():
            vx = kp * (objetivo[0] - mocap_pose['x'])
            vy = kp * (objetivo[1] - mocap_pose['y'])
            vz = kp_z * (objetivo[2] - mocap_pose['z'])
            h = np.hypot(vx, vy)
            if h > v_max:
                vx *= v_max / h
                vy *= v_max / h
            enviar(vx, vy, float(np.clip(vz, -0.3, 0.3)))
            time.sleep(periodo)

    try:
        despegar(float(altura))
        if stop_event.is_set():
            return True
        ir_al_inicio(path[0])
        if stop_event.is_set():
            return True

        # Asentarse antes de arrancar: entrar al Pure Pursuit todavía en
        # movimiento es una de las causas de sobreimpulso en el primer tramo.
        print('Estabilizando sobre el punto inicial (1.0 s)...')
        mantener(path[0], 1.0)
        if stop_event.is_set():
            return True

        pp.reset()
        pp.step(mocap_pose['x'], mocap_pose['y'])
        led_on(scf)

        # ── LAZO DE CONTROL ──
        print(f'Siguiendo la ruta ({t_est:.0f} s estimados)...')
        VENTANA_RUTA['ini'] = time.time()

        v_cmd     = float(v_prof[0])
        vz_cmd    = 0.0
        yaw_rate  = 0.0
        yaw_ref   = mocap_pose.get('yaw', 0.0)
        yaw_off   = np.radians(PP_YAW_OFFSET)
        periodo   = 1.0 / PP_CONTROL_HZ
        timeout   = PP_TIMEOUT_FACTOR * t_est + 8.0

        registro, n_ciclos, n_stale = [], 0, 0
        t_ini = time.perf_counter()
        t_next = t_ini
        t_print = t_ini

        while not stop_event.is_set():
            t_now = time.perf_counter()
            if t_now - t_ini > timeout:
                print('⚠ Timeout de vuelo — abortando el seguimiento.')
                break

            # PERRO GUARDIÁN. Con velocity setpoints el dron mantiene el
            # último comando indefinidamente; si el MoCap se queda viejo,
            # seguir empujando es volar a ciegas.
            if t_now - _t_mocap_fresco[0] > PP_MOCAP_TIMEOUT:
                n_stale += 1
                enviar(0.0, 0.0, 0.0, 0.0)
                time.sleep(periodo)
                continue

            x, y, z = mocap_pose['x'], mocap_pose['y'], mocap_pose['z']

            pp.step(x, y, max_avance=0.06 + 2.5 * v_cmd * periodo)
            if pp.finished:
                break

            v_obj = float(np.interp(pp.s_proj, pp.s, v_prof))
            v_cmd = PP_ALPHA_V * v_obj + (1.0 - PP_ALPHA_V) * v_cmd

            kappa = pp.curvatura_aqui()
            pp.lookahead_adaptativo(v_cmd, kappa)
            _, T_look, _ = pp.estado_lookahead()

            e_lat = pp.e_lat
            vx = v_cmd * T_look[0] + PP_KP_LAT * e_lat[0]
            vy = v_cmd * T_look[1] + PP_KP_LAT * e_lat[1]
            nv = np.hypot(vx, vy)
            if nv > PP_V_XY_MAX:
                vx *= PP_V_XY_MAX / nv
                vy *= PP_V_XY_MAX / nv

            z_ref  = float(pp.proj[2])
            vz_raw = v_cmd * T_look[2] + PP_KP_Z * (z_ref - z)
            vz_cmd = PP_ALPHA_VZ * vz_raw + (1.0 - PP_ALPHA_VZ) * vz_cmd
            vz     = float(np.clip(vz_cmd, -PP_VZ_MAX_CMD, PP_VZ_MAX_CMD))

            if PP_YAW_MODE == 'path':
                raw  = np.arctan2(T_look[1], T_look[0]) + yaw_off
                dif  = (raw - yaw_ref + np.pi) % (2 * np.pi) - np.pi
                yaw_ref += PP_YAW_ALPHA_REF * dif
                err = (yaw_ref - mocap_pose.get('yaw', yaw_ref)
                       + np.pi) % (2 * np.pi) - np.pi
                if abs(err) < np.radians(PP_YAW_DEADBAND):
                    err = 0.0
                cru = np.clip(PP_KP_YAW * err, -PP_YAW_RATE_MAX, PP_YAW_RATE_MAX)
                yaw_rate = (PP_YAW_ALPHA_OUT * cru
                            + (1.0 - PP_YAW_ALPHA_OUT) * yaw_rate)
            else:
                yaw_rate = 0.0

            enviar(vx, vy, vz, np.degrees(yaw_rate))
            n_ciclos += 1
            registro.append((t_now - t_ini, pp.e_lat_signed, z_ref - z,
                             v_cmd, pp.L, kappa))

            # Un print por ciclo cuesta varios ms en la consola de Windows
            # y degrada la frecuencia real del lazo. Cada 1.5 s basta.
            if t_now - t_print > 1.5:
                print(f'  [{pp.progreso():5.1f}%] v={v_cmd:.2f} L={pp.L:.2f} '
                      f'k={kappa:.2f} | e_lat={pp.e_lat_signed*100:+6.1f} cm  '
                      f'e_z={(z_ref-z)*100:+6.1f} cm')
                t_print = t_now

            t_next += periodo
            dorm = t_next - time.perf_counter()
            if dorm > 0:
                time.sleep(dorm)
            else:
                t_next = time.perf_counter()

        VENTANA_RUTA['fin'] = time.time()
        frenar()

        t_vuelo = time.perf_counter() - t_ini
        hz_real = n_ciclos / t_vuelo if t_vuelo > 0 else 0.0
        print(f'\n✓ Seguimiento terminado en {t_vuelo:.1f} s '
              f'({hz_real:.0f} Hz reales, {pp.progreso():.0f} % de la ruta)')
        if hz_real < 0.8 * PP_CONTROL_HZ:
            print(f'  ⚠ El lazo no alcanzó los {PP_CONTROL_HZ:.0f} Hz pedidos.')
            print( '    La cámara y MiDaS le están quitando CPU. Sube')
            print( '    VISION_MIN_PERIODO o baja PP_CONTROL_HZ a 30.')
        if n_stale:
            print(f'  ⚠ {n_stale} ciclos con el MoCap viejo (>{PP_MOCAP_TIMEOUT*1000:.0f} ms):')
            print( '    en esos se mandó velocidad cero por seguridad.')

        if registro:
            r = np.array(registro)
            el = np.abs(r[:, 1])
            print()
            print(f'  Error lateral : medio {el.mean()*100:5.2f} cm | '
                  f'máx {el.max()*100:5.2f} cm | p95 {np.percentile(el,95)*100:5.2f} cm')
            ez = np.abs(r[:, 2])
            print(f'  Error altura  : medio {ez.mean()*100:5.2f} cm | '
                  f'máx {ez.max()*100:5.2f} cm')
    finally:
        led_off(scf)

    if stop_event.is_set():
        frenar()
        return True

    # ── Descenso controlado manteniendo XY ──
    # OTRO FALLO MÍO, corregido: yo descendía "mientras z > LAND_HEIGHT",
    # y LAND_HEIGHT vale 0.0 en este archivo. O sea que seguía mandando
    # descenso con el dron ya en el suelo, empujando contra él hasta que
    # saltara el timeout. El original corta en z_stop = 0.10 m y deja que
    # caiga los últimos centímetros.
    print('Aterrizando suavemente...')
    t0, kp, z_stop = time.time(), 1.5, 0.10
    while mocap_pose['z'] > z_stop and not stop_event.is_set():
        if time.time() - t0 > 15.0:
            print('  ⚠ Timeout en el descenso.')
            break
        vx = kp * (path[-1][0] - mocap_pose['x'])
        vy = kp * (path[-1][1] - mocap_pose['y'])
        h = np.hypot(vx, vy)
        if h > 0.3:
            vx *= 0.3 / h
            vy *= 0.3 / h
        enviar(vx, vy, -0.15)
        time.sleep(periodo)
    frenar(10)
    try:
        cmd.send_stop_setpoint()
    except Exception:
        pass
    print('✓ En el suelo.')
    return True


def ejecutar_ruta_trayectoria(scf, ruta_suave_2d, cultivo):
    """
    Vuelo CONTINUO: sube una trayectoria polinómica al dron y la arranca.

    Diferencia con `ejecutar_ruta_fisica`: allí cada waypoint es un `go_to`
    que planifica llegar con velocidad cero, y por eso el dron frena 31 o 44
    veces. Aquí la ruta entera es un solo polinomio por tramos con velocidad
    y aceleración continuas, que el firmware evalúa a 500 Hz. El dron no
    frena en ningún punto interior, y durante el recorrido no se manda ni un
    paquete de radio: el vuelo deja de depender del temporizador de Python.

    Si algo falla —no hay módulo, no cabe en memoria, la desviación es
    inaceptable— devuelve False y el llamador vuelve al método de siempre.
    """
    from cflib.crazyflie.mem import MemoryElement

    altura   = cultivo['altura']
    land_dur = cultivo['land_dur']

    if not _HAY_TRAYECTORIA:
        return False

    # ── 1) Generar y VERIFICAR en el PC, antes de tocar el dron ──
    print()
    print('─' * 62)
    print('  TRAYECTORIA CONTINUA')
    print('─' * 62)
    tray, info = trayectoria_suave.generar_verificada(
        ruta_suave_2d, altura,
        desv_objetivo_cm=TRAY_DESV_OBJETIVO_CM,
        v_crucero=TRAY_V_CRUCERO,
        a_lat_max=TRAY_A_LAT_MAX,
        a_lon_max=TRAY_A_LON_MAX,
        v_min=TRAY_V_MIN,
        yaw_rate_max=TRAY_YAW_RATE,
        yaw_activo=YAW_ACTIVO,
        yaw_offset=YAW_OFFSET)

    if tray is None or not trayectoria_suave.resumen(info):
        print('  → No se usa la trayectoria continua; se vuela con go_to.')
        print('─' * 62)
        return False
    print('─' * 62)

    # ── 2) Esperar posición válida del MoCap ──
    print('Esperando posicion inicial del MoCap...')
    elapsed = 0.0
    while mocap_pose['x'] == 0 and mocap_pose['y'] == 0 and mocap_pose['z'] == 0:
        if stop_event.is_set():
            return True
        if elapsed >= 10.0:
            print('Timeout esperando MoCap.')
            return True
        time.sleep(0.1)
        elapsed += 0.1

    # ── 3) Subir la trayectoria ANTES de despegar ──
    # Se hace en tierra a propósito: es la única transferencia pesada por
    # radio y no queremos que compita con nada mientras el dron vuela.
    try:
        piezas = tray.a_poly4d()
        mems = scf.cf.mem.get_mems(MemoryElement.TYPE_TRAJ)
        if not mems:
            print('⚠ El firmware no expone memoria de trayectorias.')
            return False
        mem = mems[0]
        mem.poly4Ds = piezas
        print(f'Subiendo {len(piezas)} tramos al dron...')
        t0 = time.time()
        mem.write_data_sync()
        scf.cf.high_level_commander.define_trajectory(
            TRAY_ID, 0, len(piezas))
        print(f'✓ Trayectoria subida y definida ({time.time()-t0:.1f} s)')
    except Exception as e:
        print(f'⚠ No se pudo subir la trayectoria: {e}')
        return False

    hl = scf.cf.high_level_commander

    try:
        # ── 4) Despegue ──
        print(f'Despegando a {altura:.2f} m...')
        takeoff(scf, height=altura, duration=TAKEOFF_DUR)
        if not interruptible_sleep(TAKEOFF_DUR + 0.5):
            return True

        # ── 5) Ir al inicio de la trayectoria, con el rumbo ya alineado ──
        # start_trajectory usa coordenadas ABSOLUTAS: si el dron no está en
        # el punto inicial, el primer instante sería un salto.
        x0 = float(tray.tramos[0]['x'][0])
        y0 = float(tray.tramos[0]['y'][0])
        yaw0 = float(tray.tramos[0]['yaw'][0])
        px, py = mocap_pose['x'], mocap_pose['y']
        dist0 = float(np.hypot(x0 - px, y0 - py))
        t_ap = max(dist0 / VEL_APROX, 2.5)
        print(f'Aproximando al inicio: {dist0:.2f} m, rumbo '
              f'{np.degrees(yaw0):.0f}° ({t_ap:.1f} s)...')
        hl.go_to(x0, y0, altura, yaw0, t_ap, relative=False)
        if not interruptible_sleep(t_ap + 1.0):
            return True

        led_on(scf)

        # ── 6) Arrancar la trayectoria ──
        print(f'Volando la trayectoria continua: {info["duracion"]:.1f} s '
              f'a {info["v_media"]:.2f} m/s medios...')
        VENTANA_RUTA['ini'] = time.time()
        hl.start_trajectory(TRAY_ID, time_scale=1.0,
                            relative_position=False, relative_yaw=False)

        # Durante el recorrido NO se manda nada por radio. Solo se espera,
        # comprobando el paro de emergencia.
        t_fin = time.time() + info['duracion'] + 1.0
        while time.time() < t_fin:
            if stop_event.is_set():
                try:
                    hl.stop()      # aborta la trayectoria en el firmware
                except Exception:
                    pass
                return True
            restante = t_fin - time.time()
            if int(restante) % 5 == 0 and restante > 1:
                pass
            time.sleep(0.05)
        VENTANA_RUTA['fin'] = time.time()
        print('✓ Trayectoria completada.')
    finally:
        led_off(scf)

    # ── 7) Aterrizaje ──
    if not stop_event.is_set():
        print('Aterrizando...')
        land(scf, height=LAND_HEIGHT, duration=land_dur)
        interruptible_sleep(land_dur + 0.5)
    return True


def ejecutar_ruta_fisica(scf, ruta_suave_2d, cultivo):
    """
    Vuelo continuo con la nariz del dron apuntando siempre hacia adelante.

    · El yaw se manda en el MISMO go_to que la posición, así el firmware
      interpola ángulo y posición con el mismo perfil suave.
    · Ángulos ACUMULADOS (sin envolver): cada incremento es < 180°, por lo
      que el dron siempre gira por el lado corto.
    · Giros grandes (cambio de surco) se hacen parado en el aire.
    · Ningún tramo exige más de YAW_RATE_MAX: si hace falta, se alarga.
    """
    from cflib.crazyflie.high_level_commander import HighLevelCommander

    altura   = cultivo['altura']
    land_dur = cultivo['land_dur']
    print(f"  Altura: {altura} m — Aterrizaje: {land_dur} s")

    # ── Waypoints y perfil de yaw ──
    wps  = preparar_waypoints(ruta_suave_2d, MIN_DIST)
    yaws = calcular_yaws(wps) if YAW_ACTIVO else np.zeros(len(wps))
    acciones = planificar_yaw(wps, yaws)

    n_rot = sum(1 for a in acciones if a['tipo'] == 'rotar')
    print(f'  Waypoints: {len(wps)}')
    resumen_plan(wps, yaws, acciones)

    # ── Esperar posición válida del MoCap ──
    print('Esperando posicion inicial del MoCap...')
    timeout, elapsed = 10.0, 0.0
    while mocap_pose['x'] == 0 and mocap_pose['y'] == 0 and mocap_pose['z'] == 0:
        if stop_event.is_set():
            return
        if elapsed >= timeout:
            print('Timeout esperando MoCap.')
            return
        time.sleep(0.1)
        elapsed += 0.1

    # ── Despegue ──
    print(f'Despegando a {altura:.2f} m...')
    takeoff(scf, height=altura, duration=TAKEOFF_DUR)
    if not interruptible_sleep(TAKEOFF_DUR + 0.5):
        return

    hl = HighLevelCommander(scf.cf)

    # ── 1) Alineación inicial: girar en el sitio hacia el primer tramo ──
    if YAW_ACTIVO:
        x0, y0 = mocap_pose['x'], mocap_pose['y']
        # No conocemos el yaw real tras el despegue → reservamos el peor caso (180°)
        t_align = max(np.pi / YAW_RATE_MAX, 2.0)
        print(f'Alineando rumbo inicial a {np.degrees(yaws[0]):.1f}° ({t_align:.1f} s)...')
        hl.go_to(x0, y0, altura, yaw=float(yaws[0]), duration_s=t_align, relative=False)
        if not interruptible_sleep(t_align + YAW_PAUSA_ROT):
            return

    # ── 2) Aproximación al primer waypoint (¡esto faltaba!) ──
    # El dron despega donde esté, que puede estar lejos de wps[0]. Antes se
    # mandaba el primer go_to con la duración mínima → tirón inicial.
    px, py = mocap_pose['x'], mocap_pose['y']
    dist0  = float(np.linalg.norm(wps[0] - np.array([px, py])))
    t0     = max(dist0 / VEL_APROX, 2.0)
    print(f'Aproximando al inicio: {dist0:.2f} m en {t0:.1f} s...')
    hl.go_to(float(wps[0][0]), float(wps[0][1]), altura,
             yaw=float(yaws[0]), duration_s=t0, relative=False)
    if not interruptible_sleep(t0 + 0.5):
        return

    # ── 3) LED ON: ya estamos en el punto inicial de la ruta ──
    led_on(scf)

    # ── 4) Recorrido ──
    # try/finally: pase lo que pase (paro de emergencia, excepción, salida
    # anticipada), el LED se apaga. Nunca se queda encendido.
    try:
        print(f'Ejecutando ruta continua ({len(wps)} wps a {FLY_VELOCITY} m/s reales)...')
        for a in acciones:
            if stop_event.is_set():
                return

            if a['tipo'] == 'rotar':
                # Rotación en el sitio, en la posición ACTUAL medida por el MoCap
                px, py = mocap_pose['x'], mocap_pose['y']
                print(f"  ROT {a['idx']+1:3d}/{len(wps)}: yaw→{np.degrees(a['yaw']):7.1f}° "
                      f"({a['dur']:.1f} s)")
                hl.go_to(px, py, altura, yaw=a['yaw'], duration_s=a['dur'], relative=False)
                if not interruptible_sleep(a['espera']):
                    return
                continue

            print(f"  WP  {a['idx']+1:3d}/{len(wps)}: ({a['x']:.3f}, {a['y']:.3f}) "
                  f"yaw={np.degrees(a['yaw']):7.1f}° espera={a['espera']:.2f}s")
            hl.go_to(a['x'], a['y'], altura, yaw=a['yaw'],
                     duration_s=a['dur'], relative=False)

            # Se envía el siguiente destino antes de llegar (vuelo continuo)
            if not interruptible_sleep(a['espera']):
                return

        # Esperar a que el dron termine de asentarse en el último waypoint
        if not interruptible_sleep(2.0):
            return
    finally:
        # ── LED OFF: ruta terminada, antes de aterrizar ──
        led_off(scf)

    # ── Aterrizaje ──
    if not stop_event.is_set():
        print('Aterrizando...')
        land(scf, height=LAND_HEIGHT, duration=land_dur)
        interruptible_sleep(land_dur + 0.5)


# ═══════════════════════════════════════════════════════════════
# BLOQUE 5 — VISUALIZACIÓN POST-VUELO
# ═══════════════════════════════════════════════════════════════

def error_seguimiento(real_xy, ruta_2d, ventana=0.25):
    """
    Error de seguimiento (cross-track): distancia de cada punto REAL volado
    a la ruta planificada. Es la medida objetiva de "qué tan fino siguió".

    Hay dos trampas al medir esto en una ruta de cobertura, y las dos me
    dieron números falsos al probarlo:

      1. Buscar "el punto más cercano de TODA la ruta" no sirve: los surcos
         están a PASO metros, así que una desviación grande empareja con el
         surco VECINO y el error sale artificialmente pequeño.
      2. Encadenar cada muestra a la anterior tampoco: en las vueltas en U
         el índice se atasca en el vértice y a partir de ahí todo queda
         desincronizado (medía 8 cm donde solo había 0.5 cm de ruido).

    Solución: anclar cada muestra por LONGITUD DE ARCO recorrida. El dron
    avanza por la ruta aproximadamente al mismo ritmo al que se mueve, así
    que la distancia acumulada real da una estimación robusta de por dónde
    va, y solo se busca el mínimo en una ventana de ±`ventana` metros
    alrededor de ese ancla. Sin encadenamiento no hay atascos ni deriva.

    Devuelve (errores_en_metros, indices_en_la_ruta_densa).
    """
    real = np.asarray(real_xy, dtype=float)
    ruta = np.asarray(ruta_2d,  dtype=float)
    if len(real) < 2:
        return np.zeros(len(real)), np.zeros(len(real), dtype=int)

    # Densificar la ruta a paso fijo → el índice es proporcional al arco
    paso_denso = 0.005
    seg = []
    for a, b in zip(ruta[:-1], ruta[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / paso_denso)))
        seg.append(a + np.linspace(0, 1, n, endpoint=False)[:, None] * (b - a))
    densa = np.vstack(seg + [ruta[-1:]])

    # Ancla: arco recorrido de verdad, reescalado a la longitud de la ruta
    s_real = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(real, axis=0), axis=1))))
    if s_real[-1] < 1e-6:
        anclas = np.zeros(len(real), dtype=int)
    else:
        anclas = (s_real / s_real[-1] * (len(densa) - 1)).astype(int)

    n_win = max(20, int(ventana / paso_denso))

    errs, idxs = [], []
    for p, a0 in zip(real, anclas):
        lo = max(0, a0 - n_win)
        hi = min(len(densa), a0 + n_win + 1)
        d = np.linalg.norm(densa[lo:hi] - p, axis=1)
        j = lo + int(np.argmin(d))
        errs.append(float(d[j - lo]))
        idxs.append(j)
    return np.array(errs), np.array(idxs)


def plot_resultado_3d(ruta_suave_2d, altura=HOVER_HEIGHT):
    if not real_trajectory:
        print('Sin datos de trayectoria real para graficar.')
        return

    real = np.array(real_trajectory)
    teo  = np.column_stack([ruta_suave_2d, np.full(len(ruta_suave_2d), altura)])

    # ══════════════════════════════════════════════════════════════
    # SEPARAR LA RUTA DE LA APROXIMACIÓN  (arreglado 17-sep-2026)
    # ══════════════════════════════════════════════════════════════
    # Este informe medía —y esta gráfica dibujaba— TODO lo que hubiera en
    # `real_trajectory`: despegue, traslado al primer punto, giro de
    # alineación, ruta y aterrizaje, del mismo color y en el mismo número.
    #
    # Mientras la aproximación duraba cuatro segundos apenas se notaba.
    # Desde que va despacio a propósito —para no dar picos de corriente—
    # dura medio minuto, y entonces la MITAD de las muestras del "error de
    # seguimiento" son de un tramo que NO ES LA RUTA: el dron yendo en
    # línea recta desde donde despegó hasta donde la ruta empieza.
    #
    # El efecto es enorme y es puramente de contabilidad. Con el MISMO
    # seguimiento, el número subió de 78 a 110 cm solo por alargar la
    # aproximación, y en la gráfica aparecía un trazo largo cruzando el
    # dibujo que hacía que la trayectoria no se pareciera a la de antes.
    #
    #   Comprobación: RMS ≈ d·sqrt(f), con d la distancia de aproximación
    #   y f la fracción de muestras que son aproximación. Con d≈1.5 m y
    #   f = 49.4/87 = 0.57 sale 1.13 m; se midió 1.10 m.
    #
    # El seguimiento REAL de ese mismo vuelo fue de 2.68 cm RMS, el mejor
    # de los tres. Ahora la aproximación se dibuja aparte, en gris, y no
    # entra en el número.
    ini, fin = VENTANA_RUTA['ini'], VENTANA_RUTA['fin']
    t_all = np.asarray(tiempos_trayectoria, dtype=float)
    n = min(len(t_all), len(real))
    hay_ventana = (ini is not None and fin is not None and n >= 50)
    if hay_ventana:
        t_all, real_n = t_all[:n], real[:n]
        en_ruta = (t_all >= ini) & (t_all <= fin)
        hay_ventana = bool(en_ruta.sum() >= 50)
    if hay_ventana:
        antes   = real_n[t_all < ini]    # despegue + aproximación + giro
        ruta    = real_n[en_ruta]        # LA RUTA
        despues = real_n[t_all > fin]    # aterrizaje
        vuelo   = ruta
    else:
        # Sin ventana (vuelo interrumpido, o registro incompleto): se hace
        # como antes, pero avisando de que el número está sucio.
        en_vuelo = real[:, 2] > 0.6 * altura
        if en_vuelo.sum() < 10:
            en_vuelo = np.ones(len(real), dtype=bool)
        antes = despues = real[:0]
        ruta = vuelo = real[en_vuelo]

    # ── Medir el error de seguimiento ──
    errs, _ = error_seguimiento(vuelo[:, :2], ruta_suave_2d)
    rms  = float(np.sqrt(np.mean(errs ** 2)))
    p95  = float(np.percentile(errs, 95))
    emax = float(errs.max())

    print('\n' + '─' * 46)
    print('  ERROR DE SEGUIMIENTO (real vs planificada)')
    print('─' * 46)
    print(f'  RMS        : {rms*100:6.2f} cm')
    print(f'  Percentil 95: {p95*100:6.2f} cm')
    print(f'  Máximo     : {emax*100:6.2f} cm')
    print(f'  Muestras en la ruta: {len(vuelo)}')
    if hay_ventana:
        print(f'  Fuera del número: {len(antes)} muestras de despegue y')
        print(f'  aproximación y {len(despues)} de aterrizaje. No son la')
        print('  ruta, y contarlas inflaba esto hasta el metro.')
    else:
        print('  ⚠ Sin ventana de ruta: este número INCLUYE la aproximación')
        print('    y no es comparable con el de otros vuelos.')
    print(f'  Config: v={FLY_VELOCITY} m/s, MIN_DIST={MIN_DIST}, '
          f'SOLAPAMIENTO={SOLAPAMIENTO}')
    print('─' * 46)
    print('  Anota estos números al cambiar parámetros: así comparas')
    print('  configuraciones con datos en vez de a ojo.')

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle('Resultado del vuelo — Ruta teórica vs Trayectoria real', fontsize=13)

    def _trazos(ax, tridi):
        col = (lambda a: (a[:, 0], a[:, 1], a[:, 2])) if tridi \
            else (lambda a: (a[:, 0], a[:, 1]))
        ax.plot(*col(teo), 'g--', linewidth=1.2, label='Teórica')
        if len(antes):
            ax.plot(*col(antes), color='0.65', linewidth=1.0, linestyle=':',
                    label='Aproximación (no es la ruta)')
        if len(despues):
            ax.plot(*col(despues), color='0.80', linewidth=1.0, linestyle=':',
                    label='Aterrizaje')
        ax.plot(*col(ruta), 'b', linewidth=1.4, label='Ruta real (MoCap)')

    ax3 = fig.add_subplot(121, projection='3d')
    _trazos(ax3, True)
    ax3.scatter(*teo[0],  color='green',  s=60, label='Inicio')
    ax3.scatter(*teo[-1], color='orange', s=60, label='Fin')
    ax3.set_xlabel('X (m)'); ax3.set_ylabel('Y (m)'); ax3.set_zlabel('Z (m)')
    ax3.legend(fontsize=8); ax3.set_title('Vista 3D')

    ax2 = fig.add_subplot(122)
    _trazos(ax2, False)
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_title(f'Vista superior (X-Y) — ruta: {rms*100:.1f} cm RMS')

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    global scf_global, puntos_clicks

    while True:
        # ── Menú de inicio ──
        opcion, puntos_cargados, ruta_cargada = _menu_inicio_tk()

        if opcion == 'salir':
            print('Saliendo.')
            return

        if opcion == 'cargar':
            ruta_suave      = ruta_cargada
            puntos_poligono = [tuple(p) for p in puntos_cargados]
            poly_cerrado    = puntos_poligono + [puntos_poligono[0]]
        else:
            # Dibujar nuevo polígono
            puntos_clicks = []
            print("Dibuja el área de cobertura")
            dibujar_poligono()
            puntos_poligono = puntos_clicks
            poly_cerrado    = puntos_poligono + [puntos_poligono[0]]
            poligono        = Polygon(puntos_poligono)
            print("Generando ruta...")
            ruta_orig  = generar_ruta(poligono, PASO)
            ruta_suave = suavizar_bspline(ruta_orig)
            print(f"  Waypoints: {len(ruta_suave)}")

        poly_np = np.array(poly_cerrado)

        # ── Simulación ──
        accion, ruta_final = simular_y_mostrar(ruta_suave, poly_np)

        if accion == "cancelar":
            print("Cancelado.")
            return
        elif accion == "nueva":
            continue   # volver al menú
        elif accion == "volar":
            break      # ir al vuelo físico

    # ── Selección de cultivo (antes de conectar) ──
    cultivo = seleccionar_cultivo_tk()
    if cultivo is None:
        print('Vuelo cancelado — no se seleccionó cultivo.')
        return
    print(f"✓ Cultivo: {cultivo['nombre']} — Altura: {cultivo['altura']} m")

    # ── Resumen de configuración ──
    # Son muchos interruptores y es fácil volar con uno mal puesto sin
    # enterarse hasta ver el informe final. Esto lo pone todo a la vista
    # antes de armar motores.
    print()
    print('─' * 62)
    print('  CONFIGURACIÓN DE ESTE VUELO')
    print('─' * 62)
    print(f'  Ejecución     : {MODO_EJECUCION.upper()}'
          + (f'  [{TRAY_V_CRUCERO} m/s crucero, a_lat {TRAY_A_LAT_MAX}, '
             f'v_min {TRAY_V_MIN}]' if MODO_EJECUCION == 'trayectoria' else
             f'  [{FLY_VELOCITY} m/s, frena en cada waypoint]'))
    print(f'  Orientación   : {"SÍ, el EKF corrige el yaw" if USAR_ORIENTACION_MOCAP else "NO — el yaw deriva (afecta a la precisión)"}')
    print(f'  extPosStdDev  : {EXT_POS_STD_DEV if EXT_POS_STD_DEV is not None else "sin tocar"}'
          '   (↑ filtra más, ↓ salta más)')
    print(f'  MoCap → EKF   : {"COMPATIBLE (como el original)" if MODO_COMPATIBLE_ORIGINAL else "MODIFICADO"}'
          f'  [filtro {MODO_FILTRO_TS}, límite {MOCAP_HZ_MAX or "sin"} Hz]')
    print(f'  Trayectoria   : {"completa (~85 Hz)" if REGISTRAR_TRAYECTORIA_COMPLETA else "solo lo enviado"}')
    print(f'  Batería       : {"monitor ON a " + str(1000//BATERIA_PERIODO_MS) + " Hz" if MONITOR_BATERIA else "SIN monitor"}')
    print(f'  Cámara        : {"ON " + VISION_IP if VISION_ACTIVA else "OFF"}')
    if VISION_ACTIVA:
        hilos = VISION_HILOS_TORCH or max(1, (os.cpu_count() or 4) // 2)
        print(f'  MiDaS         : {"ON, " + str(hilos) + " hilos torch" if VISION_MIDAS else "OFF"}')
        print(f'  Vuelo         : {FLY_VELOCITY} m/s, altura según cultivo, '
              f'yaw {"tangente" if YAW_ACTIVO else "fijo"}')
    print(f'  LED deck      : {"ON brillo " + str(LED_BRILLO) if LED_ACTIVO else "OFF"}')
    print('─' * 62)
    print('  Antes de despegar: batería cargada, dron apagado y encendido')
    print('  (si la cámara falló en el vuelo anterior, el GAP8 sigue colgado).')
    print('─' * 62)
    print()

    # ── Vuelo físico ──
    print("Vuelo físico")
    cflib.crtp.init_drivers()

    mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
    mqtt_thread.start()
    stop_thread = threading.Thread(target=emergency_listener, daemon=True)
    stop_thread.start()

    # Hilo que manda la pose al EKF a ritmo fijo, desacoplado de MQTT.
    if MOCAP_MODO_ENVIO == 'hilo':
        extpos_thread = HiloExtpos(MOCAP_HZ_MAX if MOCAP_HZ_MAX > 0 else 100.0)
        extpos_thread.start()
        print(f'✓ Envío al EKF en hilo propio a {MOCAP_HZ_MAX:.0f} Hz')

    # Aviso temprano: sin MoCap no hay vuelo, y más vale saberlo AHORA que
    # tras esperar el timeout con el dron ya conectado.
    time.sleep(2.0)
    if not mqtt_estado['conectado']:
        print('\n⚠ ATENCIÓN: el MoCap no está conectado todavía.')
        print('  El vuelo va a fallar con "Timeout esperando MoCap" salvo que')
        print('  se reenganche en los próximos segundos. Mira el diagnóstico')
        print('  de red de arriba antes de seguir.\n')

    print("Conectando al Crazyflie...")
    scf_global = connect(URI)
    time.sleep(3.0)          # da tiempo a que baje el TOC de parámetros

    # Cuánto se fía el EKF de la posición externa. Con cadencias altas hay
    # que subirlo, o el EKF salta a cada medida retrasada y oscila.
    if EXT_POS_STD_DEV is not None:
        try:
            scf_global.cf.param.set_value('locSrv.extPosStdDev',
                                          str(EXT_POS_STD_DEV))
            print(f'✓ locSrv.extPosStdDev = {EXT_POS_STD_DEV} '
                  '(el EKF filtra la posición externa en vez de saltar a ella)')
        except Exception as e:
            print(f'⚠ No se pudo fijar extPosStdDev: {e}')

    if CONTROLADOR_ESTABILIZADOR is not None:
        try:
            scf_global.cf.param.set_value('stabilizer.controller',
                                          str(CONTROLADOR_ESTABILIZADOR))
            print(f'✓ stabilizer.controller = {CONTROLADOR_ESTABILIZADOR}'
                  + (' (PID)' if CONTROLADOR_ESTABILIZADOR == 1 else ''))
        except Exception as e:
            print(f'⚠ No se pudo fijar stabilizer.controller: {e}')

    # Si mandamos orientación, decirle al EKF cuánto fiarse de ella.
    if USAR_ORIENTACION_MOCAP:
        try:
            scf_global.cf.param.set_value('locSrv.extQuatStdDev',
                                          str(EXT_QUAT_STD_DEV))
            print(f'✓ locSrv.extQuatStdDev = {EXT_QUAT_STD_DEV} '
                  '(el EKF corrige también el yaw)')
        except Exception as e:
            print(f'⚠ No se pudo fijar extQuatStdDev: {e}')

    led_init(scf_global)     # brillo + LED apagado antes de despegar

    # ── Monitor de batería ──
    # Solo escucha (log de pm.vbat). Sirve para cruzar los reinicios del
    # ESP32 con los hundimientos de tensión durante el vuelo.
    bateria = MonitorBateria()
    try:
        bateria.iniciar(scf_global)
    except Exception as e:
        print(f'⚠ Monitor de batería no disponible: {e}')

    # Monitor del estimador: separa el error de estimación del de control.
    estimador_mon = MonitorEstimador()
    try:
        estimador_mon.iniciar(scf_global)
    except Exception as e:
        print(f'⚠ Monitor del estimador no disponible: {e}')

    # ── Cámara AI-deck + MiDaS ──
    # Se arranca DESPUÉS de conectar el dron y ANTES de despegar. Es
    # independiente del vuelo: si no conecta, se avisa y se vuela igual.
    # Sus hilos son daemon y se apagan en el finally.
    vision = SistemaVision()
    try:
        vision.iniciar()
    except Exception as e:
        print(f'⚠ Visión no disponible: {e} — el vuelo continúa.')

    # ── Barrera de batería: no despegar con la LiPo agotada ──
    if not comprobar_bateria_para_volar():
        try:
            bateria.detener()
        except Exception:
            pass
        try:
            vision.detener()
        except Exception:
            pass
        led_off(scf_global)
        disconnect(scf_global)
        scf_global = None
        print('Vuelo cancelado. Carga la bateria y vuelve a intentarlo.')
        return

    try:
        hecho = False
        if MODO_EJECUCION == 'pursuit':
            hecho = ejecutar_ruta_pursuit(scf_global, ruta_final, cultivo)
            if not hecho:
                print('→ Volviendo al método go_to para este vuelo.')
        elif MODO_EJECUCION == 'trayectoria':
            hecho = ejecutar_ruta_trayectoria(scf_global, ruta_final, cultivo)
            if not hecho:
                print('→ Volviendo al método go_to para este vuelo.')
        if not hecho:
            ejecutar_ruta_fisica(scf_global, ruta_final, cultivo)
    except Exception as e:
        print(f"Error durante vuelo: {e}")
        _cortar_motores()
    finally:
        try:
            bateria.detener()       # antes de desconectar la radio
        except Exception:
            pass
        try:
            estimador_mon.detener()
        except Exception:
            pass
        led_off(scf_global)         # red de seguridad, por si algo se saltó
        if not stop_event.is_set():
            land(scf_global)
        disconnect(scf_global)      # ← antes se ponía a None ANTES de desconectar
        scf_global = None
        try:
            vision.detener()        # cerrar cámara y ventana al terminar
        except Exception:
            pass

    try:
        estimador_mon.informe(cultivo['altura'])
    except Exception as e:
        print(f'[est] No se pudo generar el informe: {e}')

    try:
        informe_seguimiento_ventana(ruta_final)
    except Exception as e:
        print(f'[seg] No se pudo acotar el error a la ruta: {e}')

    plot_resultado_3d(ruta_final, cultivo['altura'])
    print("Mision completada.")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════
# NOTAS
# ═══════════════════════════════════════════════════════════════
#
# PRIMER VUELO (no te saltes pasos):
#   1. YAW_ACTIVO = False  → confirma que todo sigue igual que antes.
#   2. YAW_ACTIVO = True, FLY_VELOCITY = 0.05, YAW_RATE_MAX = np.radians(30),
#      área pequeña (2-3 surcos). Observa si oscila al girar.
#   3. Sube YAW_RATE_MAX de 10 en 10 grados hasta 45-60.
#
# VELOCIDAD vs PRECISIÓN — de dónde viene realmente el error
#   Hay DOS fuentes, y solo una depende de la velocidad:
#
#   (a) Geométricas, INDEPENDIENTES de la velocidad:
#       · Error de cuerda: la poligonal de waypoints no es la spline.
#         Con MIN_DIST 0.20 llegaba a 6.8 cm en las curvas; con 0.12, a 4 cm.
#       · Retraso por solapamiento: el dron vuela SOLAPAMIENTO × MIN_DIST
#         por detrás de su consigna. Con 0.40 × 0.20 eran 8 cm fijos.
#       Estas dos sumaban ~9 cm y estaban ahí TAMBIÉN a baja velocidad.
#       Corregirlas (MIN_DIST 0.12, SOLAPAMIENTO 0.30) baja el error
#       geométrico a ~4 cm y no cuesta tiempo: la ruta tarda casi lo mismo,
#       porque tramos más cortos giran menos y frenan menos en las curvas.
#
#   (b) De control, SÍ dependen de la velocidad:
#       · Retraso del controlador de posición (crece con v).
#       · Latencia MoCap → MQTT → radio → EKF: a 0.15 m/s, 100 ms de
#         latencia son 1.5 cm de posición vieja; a 0.08 m/s, 0.8 cm.
#       Estas solo se arreglan bajando la velocidad o acelerando el lazo
#       del MoCap. Mira el "[MoCap] xx Hz": si va por debajo de 30 Hz,
#       arregla eso ANTES de bajar la velocidad.
#
#   MIDE, NO ADIVINES. Al terminar el vuelo se imprime el error real
#   (RMS / P95 / máximo en cm) y una vista donde se ve DÓNDE se desvía.
#   Anota esos números al cambiar un parámetro y compara.
#
# CÓMO IR MÁS RÁPIDO (en este orden)
#   1. FLY_VELOCITY. Es el mando principal y ahora es lineal: subir de 0.08
#      a 0.15 casi parte el tiempo a la mitad. Rango sensato 0.08-0.25.
#   2. VEL_FACTOR_GIRO. Cuánto frena en las curvas. 0.6 → 0.7 recorta ~7%.
#      Si a 0.7 la nariz empieza a quedarse atrás en las curvas, bájalo.
#   3. YAW_RATE_MAX. Solo hace falta si el resumen dice que hay tramos
#      "limitados por yaw". Es un techo de seguridad, no un objetivo.
#   4. PASO. Es el que más recorta y no toca el control: separa más los
#      surcos, la ruta se acorta proporcionalmente. Pero cubre menos denso,
#      así que decídelo por el ancho de riego, no por el reloj.
#
#   El script imprime al planificar: metros, tiempo, m/s medios y cuántos
#   tramos limita cada restricción. Si dice "limitados por yaw: 0", tienes
#   margen y puedes subir FLY_VELOCITY sin que la nariz se quede atrás.
#
#   Lo que NO hay que tocar para ganar tiempo: SOLAPAMIENTO. Su efecto
#   geométrico (cuánto recorta las curvas) no depende de la velocidad,
#   porque la distancia entre waypoints es fija. Subirlo solo hace que
#   vuele más "por dentro" de la ruta.
#
# SÍNTOMAS → QUÉ TOCAR
#   Va demasiado rápido ....................... ↓ FLY_VELOCITY (ahora sí es la real)
#   Se detiene demasiadas veces a girar ....... ↑ YAW_UMBRAL_ROT
#   Mira en diagonal en las curvas ............ ↓ YAW_VENTANA (1 = sin suavizar),
#                                               ↓ VEL_FACTOR_GIRO, ↓ SOLAPAMIENTO
#   Apunta hacia atrás de donde va ............ ↑ YAW_LOOKAHEAD a 2
#   Recorta las curvas / vuela "por dentro" ... ↓ SOLAPAMIENTO
#   Tirón al empezar la ruta .................. ↓ VEL_APROX, ↑ el 2.0 mínimo de t0
#   Se desvía lateralmente al girar ........... ↓ FLY_VELOCITY, USAR_ORIENTACION_MOCAP
#
# LOS DOS TIEMPOS (importante para entender el ajuste)
#   'espera' fija la velocidad real: v = distancia / espera.
#   'dur' es lo que dura el go_to, siempre mayor, para que el dron no frene
#   en cada waypoint. Antes 'dur' hacía las dos cosas y con SOLAPAMIENTO=0.6
#   el dron volaba a 0.25 m/s cuando FLY_VELOCITY decía 0.10.
#
# FRECUENCIA DEL MOCAP  ── RESUELTO, LEE ESTO ──
#   El script imprime "[MoCap] xx Hz" cada 5 s. Por debajo de ~30 Hz el EKF
#   navega a ciegas entre correcciones y el dron oscila, sobre todo al girar.
#
#   QUÉ PASABA (medido con diagnostico.py, no supuesto):
#     Mensajes recibidos     : 1702  (85.1 Hz)   ← el MoCap iba perfecto
#     Aceptados por el filtro:   21  ( 1.1 Hz)   ← el EKF solo veía esto
#   El publicador manda el campo `ts` con resolución de UN SEGUNDO. El
#   filtro original exigía `ts` estrictamente creciente, así que de los ~85
#   mensajes de cada segundo pasaba uno y se tiraban 84. No era la red, ni
#   la cámara, ni Motive: era el filtro.
#
#   ARREGLO: MODO_FILTRO_TS = 'tolerante' acepta `ts` igual al anterior y
#   solo rechaza los genuinamente viejos. Con resolución de segundo es todo
#   lo que se puede filtrar, y deja pasar el flujo completo.
#   MOCAP_HZ_MAX = 50 limita lo que se manda al EKF: por encima de ~50 Hz
#   no mejora el estimador y sí ocupa tiempo de aire que compite con los
#   go_to. Con 85 Hz de entrada, al EKF llegan ~42 Hz.
#
#   ARREGLO DE FONDO (mejor, si controlas el publicador): que Node-RED mande
#   el `ts` con milisegundos, en ISO con decimales. Por ejemplo, en un nodo
#   function:
#       msg.payload.ts = new Date().toISOString();   // ya trae .SSS
#   Con eso puedes volver a MODO_FILTRO_TS = 'estricto' y tener además
#   protección real contra mensajes desordenados.
#
#   COMPROBACIÓN: `python diagnostico.py mocap` imprime recibidos contra
#   aceptados. Si los dos números se parecen, el filtro ya no estorba.
#
# DERIVA DEL YAW (importante)
#   Con `set_position` solo mandas POSICIÓN. El yaw del EKF viene de integrar
#   el giroscopio y DERIVA. El controlador usa ese yaw para convertir el error
#   de posición (mundo) en roll/pitch (cuerpo): si está desfasado, el dron
#   empuja torcido. Con yaw fijo casi no se nota; girando, sí.
#   Para arreglarlo pon USAR_ORIENTACION_MOCAP = True, y antes:
#     1. Comprueba que Motive exporte Z-up (o convierte el cuaternión).
#     2. Verifica `stabilizer.estimator` = 2 (Kalman) en el cfclient.
#     3. Prueba SIN volar: levanta el dron a mano, gíralo 90° y mira que
#        `stateEstimate.yaw` siga el giro real.
#     4. Recomendado: scf.cf.param.set_value('locSrv.extQuatStdDev', '0.05')
#
# COLOR LED DECK
#   Secuencia: se inicializa apagado al conectar, se enciende al llegar al
#   primer waypoint de la ruta, y se apaga al terminar, antes de aterrizar.
#   El apagado está en un `finally`, así que también ocurre si hay paro de
#   emergencia o una excepción a mitad de ruta.
#   Si el deck no está montado, avisa una vez y el vuelo sigue igual.
#   Cambia LED_COLOR (formato 0xWWRRGGBB) para otro color.
#   Los `param.set_value` mandan tráfico por radio, así que solo se llaman
#   con el dron quieto (al inicio y al final), nunca durante el recorrido.
#
# PARA LA CÁMARA / MiDaS
#   · YAW_OFFSET compensa el montaje de la cámara (si va girada 90°, pon
#     np.radians(-90)).
#   · MiDaS da profundidad RELATIVA, no métrica: escálala con la altura de
#     vuelo conocida antes de usarla como distancia a un obstáculo.
#   · Para reaccionar a un obstáculo, interrumpe el bucle de `acciones`:
#     hl.go_to(pos_actual, yaw_actual, 1.0) para frenar, y vuelve a llamar
#     a `planificar_yaw` con la ruta corregida.
#
# ═══════════════════════════════════════════════════════════════
# CÁMARA AI-DECK + MiDaS — LO QUE HAY QUE SABER
# ═══════════════════════════════════════════════════════════════
#
# EL PARO DE EMERGENCIA (lo más importante)
#   Hay DOS sitios donde funciona, y da igual cuál tenga el foco:
#     · Consola  → 'q'            (hilo `emergency_listener`, msvcrt)
#     · Ventana  → 'q' o ESC      (hilo `_bucle_ventana`, cv2.waitKey)
#   Los dos llaman a `activar_paro_emergencia()`, que es idempotente y
#   protegida con lock: aunque se disparen a la vez, los motores se cortan
#   una sola vez.
#   Cerrar la ventana con la X deja UNO de los dos, no ninguno: el de la
#   consola sigue vivo. Y 'v' en la consola vuelve a abrir la ventana, así
#   que también se recupera el otro. Antes, cerrar la ventana mataba su hilo
#   para siempre y no había forma de traerla de vuelta.
#
#   Por qué era un riesgo y cómo está resuelto:
#     1. Foco de ventana. Con la ventana de OpenCV activa, msvcrt.kbhit()
#        de la consola NO ve nada. → Resuelto: la ventana escucha también.
#     2. Bloqueo por inferencia. Si MiDaS corriera en el mismo hilo que
#        cv2.waitKey(), cada inferencia (100-400 ms en CPU) sería un hueco
#        en el que la tecla no se lee. → Resuelto: MiDaS vive en el hilo
#        `midas` y la ventana en el hilo `ventana-cam`. La ventana solo
#        dibuja, así que llama a waitKey ~30 veces por segundo SIEMPRE.
#     3. Hambre de CPU. Torch puede tomar todos los núcleos y dejar sin
#        turno al hilo del MoCap y al del teclado. → Resuelto con
#        VISION_HILOS_TORCH (límite de núcleos), VISION_MIN_PERIODO
#        (cadencia máxima de inferencia) y prioridad de hilo baja.
#     4. Stream caído. Si el AI-deck se desconecta, la ventana no se queda
#        congelada esperando: pinta "SIN CONEXION" y sigue leyendo teclas.
#     5. Excepciones. Todo el subsistema está envuelto en try/except; un
#        fallo de cámara imprime y sigue, nunca tumba el vuelo.
#
#   COMPROBACIÓN ANTES DE VOLAR (hazla, son 30 segundos):
#     Con el dron en el suelo y las hélices QUITADAS, arranca el script,
#     deja que aparezca la ventana de la cámara, haz clic en ella para
#     darle el foco y pulsa 'q'. Debe imprimirse
#     "🚨 PARO DE EMERGENCIA ACTIVADO [ventana de la camara]". Repite con
#     el foco en la consola.
#
# AJUSTES DE IMAGEN: EL BANCO MANDA
#   `depth_simulator/Ai deck MiDas stream.py` y este script comparten los
#   mismos parámetros, con los mismos nombres y escalas. El flujo es:
#     1. Ajustas en el banco con los deslizadores, viendo el resultado.
#     2. Sales con 'q'. El banco escribe `pruebas_camara/ultimos_ajustes.json`.
#     3. Este script lo lee al arrancar (VISION_AJUSTES_JSON) y vuela con eso.
#   No hay que copiar números a mano. Si el JSON no existe, se usan los
#   valores de VISION_AJUSTES, que ya traen tu configuración del banco.
#
#   Al arrancar verás en consola qué ajustes se cargaron, y en la ventana de
#   la cámara la línea con realce / clahe / brillo / contraste / wb / color.
#   Si esa línea no coincide con lo que esperabas, el JSON está mandando.
#
#   OJO CON EL MODO DE COLOR: solo cambia lo que VES. A MiDaS siempre se le
#   pasa la imagen tonal, nunca el índice de vegetación ni el falso color,
#   porque el modelo espera una foto de la escena y un mapa de índice lo
#   despista. Así puedes volar mirando ExG sin estropear la profundidad.
#
#   Y 'promedio' déjalo en 1 para volar. Promediar frames baja el ruido
#   muchísimo en el banco con la cámara quieta, pero en vuelo emborrona.
#
# ¿SE PUEDEN SUBIR LOS FPS?
#   Desde este script NO. El ritmo lo marca el GAP8: captura → codifica →
#   transmite, en bucle, tan rápido como pueda. No hay ningún parámetro de
#   fps ni en el ejemplo `wifi-img-streamer` ni expuesto al Crazyflie; la PC
#   solo consume lo que llegue. Para subirlos hay que quitarle trabajo al
#   GAP8 y al ESP32:
#
#   · JPEG en vez de RAW. Es LA palanca. En wifi-img-streamer.c:
#         static StreamerMode_t streamerMode = RAW_ENCODING;
#     cámbialo a JPEG_ENCODING, recompila y flashea el GAP8. Una imagen RAW
#     de 324x244 son 79 kB; en JPEG bajan a ~5-10 kB. Es ~10x menos tráfico,
#     así que suben los fps Y se estabiliza el enlace a la vez. El receptor
#     de este script ya soporta los dos formatos: no hay que tocar nada aquí.
#
#   · Después de eso, el cuello de botella pasa a ser el enlace WiFi. Si aun
#     así van justos, no hay más margen sin bajar resolución en el firmware.
#
#   OJO: subir los fps con RAW mientras el enlace ya se está cayendo es ir en
#   la dirección contraria. Más tráfico = más caídas. Primero JPEG, luego
#   mides, y solo entonces decides si necesitas más.
#
#   Y para MiDaS, más fps de cámara casi no cambia nada: VISION_MIN_PERIODO
#   limita la inferencia a 5 Hz de todas formas, y ese límite es el que
#   protege al MoCap y al paro de emergencia. No lo subas sin medir el Hz
#   del MoCap después.
#
# ¿ES DISTANCIA O ES CARGA? — QUE LO DIGAN LOS DATOS
#   Al terminar el vuelo, el script imprime `resumen_caidas()`: dónde estaba
#   el dron en CADA caída, y compara la distancia media de esos puntos con la
#   de todo el vuelo.
#     · Caídas sistemáticamente más lejos  → sí, es alcance/orientación.
#     · Caídas repartidas cerca y lejos    → no es distancia.
#   Pon VISION_POS_LAPTOP = (x, y, z) con la posición real del receptor en
#   coordenadas del MoCap para que la comparación sea buena.
#
#   Y la prueba de 2 minutos que zanja la discusión: deja el dron EN EL SUELO
#   al lado de la laptop, sin volar, con el stream corriendo 5 minutos.
#     · Si se cae ahí → no es distancia, es carga/alimentación/interferencia.
#     · Si no se cae en el suelo pero sí volando → puede ser distancia, pero
#       también la ANTENA: el ESP32 lleva una antena de PCB diminuta y este
#       script gira la nariz del dron todo el rato (YAW_ACTIVO). Una antena
#       así tiene nulos de radiación: a ciertos rumbos irradia mucho peor.
#       Si las caídas se agrupan por RUMBO y no por distancia, es eso.
#
#   Un sospechoso que la gente olvida: el nRF51 del propio Crazyflie
#   transmite a centímetros de la antena del AI-deck, y durante el vuelo va
#   a tope (extpos del MoCap a 30+ Hz más los go_to). Esa cercanía desensibiliza
#   al receptor del ESP32 por muy pegada que esté la laptop. Por eso puede
#   caerse volando y no en el banco: en el banco no hay tráfico de radio.
#   Separa las frecuencias todo lo posible: tu URI usa canal 80 = 2480 MHz,
#   así que fija el AP del deck en el CANAL 1 (2412 MHz), que es el más
#   alejado. Si el AP está en el canal 11 (2462 MHz) se están pisando.
#
# EL DATARATE DE LA RADIO: 1M vs 2M  (URI radio://0/80/1M o /2M)
#   No es un detalle menor cuando hay una cámara WiFi a 2 cm de la antena
#   del nRF51. Los dos efectos van en sentidos OPUESTOS:
#
#   · A 1M cada paquete tarda el DOBLE en el aire que a 2M. Mismo tráfico,
#     el doble de tiempo transmitiendo. Y el nRF51 del Crazyflie está a
#     centímetros de la antena del AI-deck, así que más tiempo de aire =
#     más rato desensibilizando al receptor del ESP32.
#   · Pero a 1M el canal es más estrecho (1 MHz frente a 2 MHz), así que
#     ocupa menos espectro y molesta menos por ancho de banda.
#
#   Lo que sí es objetivo: a 1M tienes la MITAD de capacidad. Con extpos del
#   MoCap a 30+ Hz, más los go_to, más el log de batería, el enlace va más
#   apretado. Si ves el "[MoCap] xx Hz" bajo, prueba 2M.
#
#   El canal 80 = 2480 MHz en ambos casos. Es el extremo alto de la banda:
#   buena separación si el AP del deck está en el canal 1 (2412 MHz), mala
#   si está en el 11 (2462 MHz). Mira en qué canal está tu AP.
#
#   Regla práctica: cambia UNA cosa a la vez y mira el contador de "caidas
#   wifi" de la ventana. Es la única forma de saber si ayudó.
#
# EL SSID DESAPARECE Y REAPARECE DURANTE EL VUELO
#   Esto NO es lo mismo que "se cae la conexión". Si la red entera se va de
#   la lista de WiFi y vuelve, el punto de acceso dejó de existir un momento:
#   EL ESP32 SE ESTÁ REINICIANDO. Eso descarta de golpe la distancia (a más
#   distancia la señal se degrada, no se reinicia el AP), descarta que sea
#   Windows saltando de red, y descarta la orientación de la antena.
#
#   Causas, por orden:
#
#   1. BROWNOUT (la más probable con diferencia). En vuelo, cada acelerón de
#      los motores hunde la tensión de la LiPo. Si en ese instante el ESP32
#      está transmitiendo, el pico de corriente se suma y el regulador se
#      queda sin margen → reset del ESP32 → el AP desaparece y vuelve.
#      Encaja con el síntoma clave: en el banco no pasa, volando sí.
#        · Batería recién cargada. Una LiPo vieja tiene mucha resistencia
#          interna y se hunde el doble bajo carga.
#        · TU CASO CONCRETO: llevas el Color LED Deck a LED_BRILLO = 255 y
#          se enciende justo al empezar la ruta, o sea en pleno vuelo. Baja
#          LED_BRILLO a 60-80, o haz un vuelo de prueba con
#          LED_ACTIVO = False. Si las caídas desaparecen, era eso.
#        · Quita los decks que no necesites para esta prueba. Menos peso =
#          menos corriente de motores = menos hundimiento.
#        · JPEG en vez de RAW: menos tiempo de radio encendida = menos pico.
#      El script ahora lo MIDE: `resumen_caidas()` imprime la tensión mínima
#      en los segundos previos a cada caída. Si se hunde en cada una, es esto.
#
#   2. CONTACTO MECÁNICO. La vibración del vuelo puede hacer que el conector
#      del AI-deck pierda contacto un instante. Revisa que esté bien asentado
#      y que los pines no estén torcidos. También encaja con "solo en vuelo".
#
#   3. WATCHDOG DEL FIRMWARE del ESP32 bajo carga. Actualiza el firmware del
#      NINA desde el cfclient, junto con cfclient y cflib.
#
#   Cómo verlo en directo desde otra ventana de PowerShell, mientras vuela:
#       while ($true) { netsh wlan show networks | Select-String "streaming";
#                       Start-Sleep -Milliseconds 500 }
#   Si la línea desaparece y vuelve, confirmado: se reinicia el ESP32.
#
# SE CAE EL WIFI DEL AI-DECK AL EMPEZAR A TRANSMITIR
#   Si el enlace se cae justo cuando arranca el stream, con la laptop AL LADO,
#   NO es distancia: es carga. Va por orden de probabilidad.
#
#   1. WINDOWS SE VA SOLO. El AP del AI-deck no tiene internet. Windows 11,
#      al detectarlo, salta a otra red guardada que sí la tenga. Parece que
#      "se cayó el deck" y en realidad se fue la laptop.
#        Configuración → Red e Internet → Wi-Fi → Administrar redes conocidas
#        → en la red del laboratorio (y en CUALQUIER otra guardada):
#          quitar "Conectarse automáticamente cuando esté dentro del alcance".
#      Y quita también el ahorro de energía del adaptador:
#        Administrador de dispositivos → adaptador Wi-Fi → Propiedades →
#        Administración de energía → desmarcar "Permitir que el equipo apague
#        este dispositivo para ahorrar energía".
#      Comprobación: con el stream corriendo, mira si la laptop terminó
#      conectada a OTRA red. Si sí, era esto.
#
#   2. EL ESP32 SE AHOGA CON RAW. Una imagen RAW son 324x244 = 79 kB. A 10
#      fps son ~6 Mbps sostenidos, que es el límite práctico del NINA en
#      modo AP, y encima es cuando más corriente pide.
#        → Cambia el streamer a JPEG (parámetro del ejemplo wifi-img-streamer).
#          Baja el tráfico ~10x y el enlace se vuelve estable.
#        → Baja los fps del GAP8 si sigues en RAW.
#
#   3. ALIMENTACIÓN. El pico de TX del ESP32 sobre una batería a media carga
#      provoca un brownout: el ESP32 se reinicia y tira el AP. Prueba con la
#      batería recién cargada, o alimentando por USB desde un puerto que dé
#      corriente de verdad (no un hub pasivo).
#
#   4. MODO ESTACIÓN en vez de AP. Es la solución de fondo y la que
#      recomienda Bitcraze cuando el AP va inestable: el deck se une al
#      router del laboratorio y la laptop NUNCA se asocia al ESP32. De paso
#      resuelve la convivencia con el MoCap. Pon VISION_IP a la IP que le dé
#      el router.
#
#   5. INTERFERENCIA CON EL CRAZYRADIO. URI radio://0/80/2M = canal 80 =
#      2480 MHz, justo encima del extremo alto de la banda WiFi 2.4 GHz.
#      Si el AP del deck cayó en el canal 11 (2462 MHz), se pisan. Separa:
#      baja el Crazyradio (radio://0/30/2M = 2430 MHz) o fija el canal del
#      AP en el 1. Si el router del laboratorio tiene 5 GHz, mete ahí la
#      laptop y deja los 2.4 GHz para el dron.
#
#   6. FIRMWARE DEL NINA (ESP32) VIEJO. Bitcraze reconoce que el streamer
#      era inestable en versiones antiguas. Actualiza el firmware del ESP32
#      desde el cfclient, junto con cfclient/cflib.
#
#   El script ya no se queda muerto si esto pasa: el receptor reintenta solo
#   (VISION_RECONECTAR) y la ventana muestra el contador "caidas wifi".
#   Si ese contador sube durante el vuelo, tienes el problema medido.
#
# RED — EL DETALLE QUE MÁS PROBLEMAS DA
#   El MoCap llega por MQTT (192.168.50.200) y la cámara por WiFi del deck.
#   Si conectas la PC al punto de acceso propio del AI-deck (192.168.4.1)
#   te quedas SIN la red del laboratorio → sin MoCap → sin vuelo.
#   Lo correcto es poner el AI-deck en modo ESTACIÓN, unido a la misma red
#   del laboratorio, y apuntar VISION_IP a la IP que reciba ahí
#   (por ejemplo VISION_IP = '192.168.50.xx'). Con eso conviven MoCap y
#   cámara en la misma interfaz.
#
# MODELO SIN INTERNET
#   La primera vez, MiDaS se descarga de internet. Con internet disponible,
#   ejecuta una vez:
#       python -c "import sistema_vuelo as p; p.precargar_midas()"
#   Después ya funciona sin conexión a internet.
#
# AJUSTE RÁPIDO
#   El dron va raro / el MoCap baja de 30 Hz .. ↑ VISION_MIN_PERIODO (0.3-0.5),
#                                               ↓ VISION_HILOS_TORCH a 1,
#                                               o VISION_MIDAS = False
#   Vídeo a tirones ........................... VISION_MODELO = 'MiDaS_small'
#   Colores raros (cielo naranja) ............. otro 'patron' en VISION_AJUSTES
#   Imagen muy oscura ......................... 'realce': 2 (CLAHE),
#                                               'clahe_clip': 30, 'clahe_tile': 8
#   No quiero cámara hoy ...................... VISION_ACTIVA = False
#
# LO QUE LA VISIÓN **NO** HACE
#   No manda ni un solo comando de vuelo. No toca la ruta, ni el yaw, ni el
#   planificador: `ejecutar_ruta_fisica` está exactamente igual que antes.
#   Lo único que la visión puede hacerle al dron es cortarle los motores.
#   Cuando quieras evasión de obstáculos, ese es el siguiente paso: leer
#   `vision.ultimo()['info']` desde el bucle de acciones.
