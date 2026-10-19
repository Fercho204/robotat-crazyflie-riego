"""
Prueba de vuelo bajo — Crazyflie 2.1 SIN FlowDeck
==================================================
Sube ~5-10 cm, hover 2 segundos, aterriza suave.
"""

import usb.backend.libusb0
import logging
import time
import threading
import msvcrt  

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

URI = "radio://0/80/2M/E7E7E7E7F9"
logging.basicConfig(level=logging.ERROR)

THRUST_TAKEOFF = 45000
THRUST_HOVER   = 40000
THRUST_LAND    = 30000
HOVER_TIME     = 2.0

# 🚨 PARO DE EMERGENCIA
emergency_stop = False

def emergency_listener():
    global emergency_stop
    print("🚨 Presiona 'q' para paro de emergencia")

    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8').lower()
            if key == 'q':
                print("🚨 PARO DE EMERGENCIA ACTIVADO")
                emergency_stop = True
                break

def ramp(cf, start, end, steps=15, delay=0.05):
    global emergency_stop

    step = (end - start) / steps
    for i in range(steps):

        # 🚨 PARO
        if emergency_stop:
            print("🚨 CORTE INMEDIATO DE MOTORES")
            cf.commander.send_setpoint(0, 0, 0, 0)
            cf.commander.send_stop_setpoint()
            return

        cf.commander.send_setpoint(0, 0, 0, int(start + step * i))
        time.sleep(delay)

if __name__ == "__main__":
    cflib.crtp.init_drivers()
    print("Conectando...")

    try:
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
            print("✓ Conectado")

            # 🚨 THREAD DE PARO
            stop_thread = threading.Thread(target=emergency_listener, daemon=True)
            stop_thread.start()

            input("\n⚠️  Dron en el suelo, espacio despejado → ENTER para volar...")

            cf = scf.cf
            cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.1)

            print("Despegando...")
            ramp(cf, 10001, THRUST_TAKEOFF)

            print(f"Hover {HOVER_TIME}s...")
            t_end = time.time() + HOVER_TIME
            while time.time() < t_end:

                # 🚨 PARO
                if emergency_stop:
                    print("🚨 PARO → corte inmediato")
                    cf.commander.send_setpoint(0, 0, 0, 0)
                    cf.commander.send_stop_setpoint()
                    break

                cf.commander.send_setpoint(0, 0, 0, THRUST_HOVER)
                time.sleep(0.05)

            print("Aterrizando...")
            ramp(cf, THRUST_HOVER, THRUST_LAND, steps=10)
            time.sleep(1.2)

            cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.1)
            cf.commander.send_stop_setpoint()
            print("✓ Prueba completada")

    except Exception as e:
        print(f"✗ Error: {e}")