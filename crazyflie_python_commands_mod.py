"""
Cambios de versión: 
    - Ahora el logconfig se crea en connect() y se guarda en el objeto scf.
    - Se agregó la inicialización del filtro de Kalman
"""

import logging
import time
import sys
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.high_level_commander import HighLevelCommander

cflib.crtp.init_drivers(enable_debug_driver=False)
logging.basicConfig(level=logging.CRITICAL)
   
def connect(uri):
    """
    Conecta al Crazyflie usando el URI proporcionado.

    Parámetros:
        uri (str): El URI del Crazyflie.

    Retorno:
        SyncCrazyflie: Un objeto de sincronización de Crazyflie si la conexión es exitosa.

    Errores:
        Imprime errores específicos si el dongle Crazyradio no está conectado o si la conexión es rechazada.
    """
    try:        
        scf = SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache'))
        scf.open_link()
        print(f"Connection to Crazyflie established successfully.")

        # --- Crear un logger global de pose ---
        pose_log_config = LogConfig(name='Pose', period_in_ms=100)
        pose_log_config.add_variable('stateEstimate.x', 'float')
        pose_log_config.add_variable('stateEstimate.y', 'float')
        pose_log_config.add_variable('stateEstimate.z', 'float')
        pose_log_config.add_variable('stateEstimate.roll', 'float')
        pose_log_config.add_variable('stateEstimate.pitch', 'float')
        pose_log_config.add_variable('stateEstimate.yaw', 'float')

        pose_data = {'x': 0, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0}

        def pose_callback(timestamp, data, logconf):
            pose_data.update({
                'x': data['stateEstimate.x'],
                'y': data['stateEstimate.y'],
                'z': data['stateEstimate.z'],
                'roll': data['stateEstimate.roll'],
                'pitch': data['stateEstimate.pitch'],
                'yaw': data['stateEstimate.yaw']
            })

        pose_log_config.data_received_cb.add_callback(pose_callback)
        scf.cf.log.add_config(pose_log_config)
        pose_log_config.start()

        # Guardar el logger y los datos en el objeto
        scf.pose_logger = pose_log_config
        scf.pose_data = pose_data

        scf.cf.param.set_value('stabilizer.estimator', '2')  # 2 = Kalman
        # --- Reiniciar el estimador Kalman ---
        print("Reiniciando el estimador Kalman...")
        scf.cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        scf.cf.param.set_value('kalman.resetEstimation', '0')  # vuelve a 0
        # Esperar a que el EKF converja antes del vuelo
        time.sleep(3.0)
        print("Estimador reiniciado correctamente.")

        return scf
    
    except Exception as e:
        if 'Cannot find a Crazyradio Dongle' in str(e):
            print(f"Error: Crazyradio Dongle not found. Ensure the dongle is connected properly.")
        elif 'Connection refused' in str(e):
            print(f"Error: Connection to Crazyflie was refused. Check if the Crazyflie is powered on and in range.")
        else:
            print(f"General error occurred while trying to connect to Crazyflie. Error details: {str(e)}")

def disconnect(scf):
    """
    Desconecta el Crazyflie.

    Parámetros:
        scf (SyncCrazyflie): Objeto SyncCrazyflie para cerrar la conexión.

    Errores:
        Imprime errores si ocurre algún problema durante la desconexión.
    """
    try:
        if scf:
            if hasattr(scf, 'pose_logger'):
                scf.pose_logger.stop()
                scf.pose_logger.delete()
            scf.close_link()
            print(f"Successfully disconnected from Crazyflie.")
        else:
            print(f"Error: Invalid SyncCrazyflie object. No connection to close.")

    except Exception as e:
        print(f"Error: An issue occurred while disconnecting from Crazyflie. Error details: {str(e)}")

def detect_flow_deck(scf):
    """
    Detecta si el Flow Deck está instalado en el Crazyflie.

    Parámetros:
        scf (SyncCrazyflie): Objeto SyncCrazyflie con la conexión establecida.

    Retorno:
        int: Retorna 1 si el Flow Deck es detectado, 0 en caso contrario.

    Errores:
        Imprime un mensaje de error si ocurre algún problema durante la detección.
    """
    try:
        flow_deck_detected = scf.cf.param.get_value('deck.bcFlow2')

        if flow_deck_detected == '1':
            print(f"Flow Deck detected successfully.")
            return 1
        else:
            print(f"Flow Deck not detected. Please verify that it is installed properly.")
            return 0
    except Exception as e:
        print(f"Error: An issue occurred while detecting the Flow Deck. Error details: {str(e)}")

def get_pose(scf):
    """
    Obtiene la posición y orientación del Crazyflie en términos de coordenadas (x, y, z) y ángulos (roll, pitch, yaw).

    Parámetros:
        scf (SyncCrazyflie): Objeto SyncCrazyflie con la conexión establecida.

    Retorno:
        list: Una lista con los valores de posición [x, y, z] y orientación [roll, pitch, yaw].

    Errores:
        Imprime un mensaje de error si ocurre algún problema al obtener la pose.
    """
    try:
        pose = scf.pose_data
        return [pose['x'], pose['y'], pose['z'], pose['roll'], pose['pitch'], pose['yaw']]
    except AttributeError:
        print("ERROR: Pose logger not initialized. Ensure connect() was called first.")

def set_position(scf, x, y, z):
    """
    Establece una posición absoluta para el Crazyflie.

    Parámetros:
        scf (SyncCrazyflie): Objeto SyncCrazyflie con la conexión establecida.
        x (float): Coordenada X.
        y (float): Coordenada Y.
        z (float): Coordenada Z.

    Errores:
        Imprime un mensaje de error si ocurre algún problema al establecer la posición.
    """
    try:
        if not all(isinstance(coord, (int, float)) for coord in [x, y, z]):
            print(f"ERROR: Input values invalids.")

        scf.cf.extpos.send_extpos(x, y, z)
        time.sleep(0.01)
        print(f"Absolute position successfully set.")

    except Exception as e:
        print(f"ERROR: An error occurred during the position update: {str(e)}")

def set_pose(scf, x, y, z, qx, qy, qz, qw):
    """
    Establece una pose absoluta en el espacio, con posición y orientación en cuaterniones.

    Parámetros:
        scf (SyncCrazyflie): Objeto SyncCrazyflie con la conexión establecida.
        x, y, z (float): Coordenadas de posición.
        qx, qy, qz, qw (float): Componentes del cuaternión de orientación.

    Errores:
        Imprime un mensaje de error si ocurre algún problema al establecer la pose.
    """
    try:
        if not all(isinstance(coord, (int, float)) for coord in [x, y, z]):
            print(f"ERROR: Input values invalids.")

        scf.cf.extpos.send_extpose(x, y, z, qx, qy, qz, qw)
        time.sleep(0.01)
        print(f"Absolute pose successfully set.")

    except Exception as e:
        print(f"ERROR: An error occurred during the pose update: {str(e)}")

def get_pid_values(scf):
    """
    Obtiene los valores de los controladores PID del Crazyflie para los ejes X, Y y Z.

    Parámetros:
        scf (SyncCrazyflie): Objeto SyncCrazyflie con la conexión establecida.

    Retorno:
        dict: Diccionario con los valores PID para cada eje (X, Y, Z).

    Errores:
        Imprime un mensaje de error si ocurre algún problema al obtener los valores PID.
    """
    try:
        pid_values = {
            'X': [
                float(scf.cf.param.get_value('posCtlPid.xKp')),
                float(scf.cf.param.get_value('posCtlPid.xKi')),
                float(scf.cf.param.get_value('posCtlPid.xKd'))
            ],
            'Y': [
                float(scf.cf.param.get_value('posCtlPid.yKp')),
                float(scf.cf.param.get_value('posCtlPid.yKi')),
                float(scf.cf.param.get_value('posCtlPid.yKd'))
            ],
            'Z': [
                float(scf.cf.param.get_value('posCtlPid.zKp')),
                float(scf.cf.param.get_value('posCtlPid.zKi')),
                float(scf.cf.param.get_value('posCtlPid.zKd'))
            ]
        }
        print("PID values for X axis: P = {:.2f}, I = {:.2f}, D = {:.2f}".format(pid_values['X'][0], pid_values['X'][1], pid_values['X'][2]))
        print("PID values for Y axis: P = {:.2f}, I = {:.2f}, D = {:.2f}".format(pid_values['Y'][0], pid_values['Y'][1], pid_values['Y'][2]))
        print("PID values for Z axis: P = {:.2f}, I = {:.2f}, D = {:.2f}".format(pid_values['Z'][0], pid_values['Z'][1], pid_values['Z'][2]))
        return pid_values
    
    except Exception as e:
        print(f"ERROR: An error occurred during the pid values lecture: {str(e)}")

def set_pid_values(scf, p_gains, i_gains, d_gains):
    """
    Establece los valores de los controladores PID para cada eje (X, Y, Z).

    Parámetros:
        scf (SyncCrazyflie): Objeto SyncCrazyflie con la conexión establecida.
        p_gains (dict): Ganancias P para los ejes X, Y y Z.
        i_gains (dict): Ganancias I para los ejes X, Y y Z.
        d_gains (dict): Ganancias D para los ejes X, Y y Z.

    Errores:
        Imprime un mensaje de error si ocurre algún problema al establecer los valores PID.
    """
    try:       
        # X Axis
        scf.cf.param.set_value('posCtlPid.xKp', p_gains['X'])
        scf.cf.param.set_value('posCtlPid.xKi', i_gains['X'])
        scf.cf.param.set_value('posCtlPid.xKd', d_gains['X'])
        
        # Y Axis
        scf.cf.param.set_value('posCtlPid.yKp', p_gains['Y'])
        scf.cf.param.set_value('posCtlPid.yKi', i_gains['Y'])
        scf.cf.param.set_value('posCtlPid.yKd', d_gains['Y'])
        
        # Z Axis
        scf.cf.param.set_value('posCtlPid.zKp', p_gains['Z'])
        scf.cf.param.set_value('posCtlPid.zKi', i_gains['Z'])
        scf.cf.param.set_value('posCtlPid.zKd', d_gains['Z'])

        print(f"Successful PID modification.")
    
    except Exception as e:
        print(f"ERROR: An error occurred during the PID modification: {str(e)}")

def get_pid_x(scf):
    """
    Obtiene los valores del controlador PID para el eje X.

    Parámetros:
        scf (SyncCrazyflie): Objeto SyncCrazyflie con la conexión establecida.

    Retorno:
        dict: Diccionario con los valores P, I y D para el eje X.

    Errores:
        Imprime un mensaje de error si ocurre algún problema al obtener los valores PID.
    """
    try:
        pid_x = {
            'P': float(scf.cf.param.get_value('posCtlPid.xKp')),
            'I': float(scf.cf.param.get_value('posCtlPid.xKi')),
            'D': float(scf.cf.param.get_value('posCtlPid.xKd'))
        }
        
        print("PID values for X axis: P = {:.2f}, I = {:.2f}, D = {:.2f}".format(pid_x['P'], pid_x['I'], pid_x['D']))
        return pid_x
    
    except Exception as e:
        print(f"ERROR: An error occurred while retrieving the PID values for X axis: {str(e)}")

def get_pid_y(scf):
    """
    Obtiene los valores del controlador PID para el eje Y.

    Parámetros:
        scf (SyncCrazyflie): Objeto SyncCrazyflie con la conexión establecida.

    Retorno:
        dict: Diccionario con los valores P, I y D para el eje Y.

    Errores:
        Imprime un mensaje de error si ocurre algún problema al obtener los valores PID.
    """
    try:
        pid_y = {
            'P': float(scf.cf.param.get_value('posCtlPid.yKp')),
            'I': float(scf.cf.param.get_value('posCtlPid.yKi')),
            'D': float(scf.cf.param.get_value('posCtlPid.yKd'))
        }
        
        print("PID values for Y axis: P = {:.2f}, I = {:.2f}, D = {:.2f}".format(pid_y['P'], pid_y['I'], pid_y['D']))
        return pid_y
    
    except Exception as e:
        print(f"ERROR: An error occurred while retrieving the PID values for Y axis: {str(e)}")

def get_pid_z(scf):
    """
    Obtiene los valores del controlador PID para el eje Z.

    Parámetros:
        scf (SyncCrazyflie): Objeto SyncCrazyflie con la conexión establecida.

    Retorno:
        dict: Diccionario con los valores P, I y D para el eje Z.

    Errores:
        Imprime un mensaje de error si ocurre algún problema al obtener los valores PID.
    """
    try:
        pid_z = {
            'P': float(scf.cf.param.get_value('posCtlPid.zKp')),
            'I': float(scf.cf.param.get_value('posCtlPid.zKi')),
            'D': float(scf.cf.param.get_value('posCtlPid.zKd'))
        }
        
        print("PID values for Z axis: P = {:.2f}, I = {:.2f}, D = {:.2f}".format(pid_z['P'], pid_z['I'], pid_z['D']))
        return pid_z
    
    except Exception as e:
        print(f"ERROR: An error occurred while retrieving the PID values for Z axis: {str(e)}")

def set_pid_x(scf, P, I, D):
    """
    Establece los valores del controlador PID para el eje X.

    Parámetros:
        scf (SyncCrazyflie): Objeto SyncCrazyflie con la conexión establecida.
        P (float): Ganancia P.
        I (float): Ganancia I.
        D (float): Ganancia D.

    Errores:
        Imprime un mensaje de error si ocurre algún problema al establecer los valores PID.
    """
    try:      
        scf.cf.param.set_value('posCtlPid.xKp', P)
        scf.cf.param.set_value('posCtlPid.xKi', I)
        scf.cf.param.set_value('posCtlPid.xKd', D)

        print(f"Successful PID modification.")
    
    except Exception as e:
        print(f"ERROR: An error occurred during the PID modification: {str(e)}")

def set_pid_y(scf, P, I, D):
    """
    Establece los valores del controlador PID para el eje Y.

    Parámetros:
        scf (SyncCrazyflie): Objeto SyncCrazyflie con la conexión establecida.
        P (float): Ganancia P.
        I (float): Ganancia I.
        D (float): Ganancia D.

    Errores:
        Imprime un mensaje de error si ocurre algún problema al establecer los valores PID.
    """
    try:    
        scf.cf.param.set_value('posCtlPid.yKp', P)
        scf.cf.param.set_value('posCtlPid.yKi', I)
        scf.cf.param.set_value('posCtlPid.yKd', D)

        print(f"Successful PID modification.")
    
    except Exception as e:
        print(f"ERROR: An error occurred during the PID modification: {str(e)}")

def set_pid_z(scf, P, I, D):
    """
    Establece los valores del controlador PID para el eje Z.

    Parámetros:
        scf (SyncCrazyflie): Objeto SyncCrazyflie con la conexión establecida.
        P (float): Ganancia P.
        I (float): Ganancia I.
        D (float): Ganancia D.

    Errores:
        Imprime un mensaje de error si ocurre algún problema al establecer los valores PID.
    """
    try:  
        scf.cf.param.set_value('posCtlPid.zKp', P)
        scf.cf.param.set_value('posCtlPid.zKi', I)
        scf.cf.param.set_value('posCtlPid.zKd', D)

        print(f"Successful PID modification.")
    
    except Exception as e:
        print(f"ERROR: An error occurred during the PID modification: {str(e)}")

def takeoff(scf, height = 0.3, duration = 1.0):
    """
    Comanda al Crazyflie a despegar a una altura especificada.

    Parámetros:
        scf (SyncCrazyflie): Objeto SyncCrazyflie con la conexión establecida.
        height (float): Altura a alcanzar en el despegue.
        duration (float): Duración del despegue en segundos.

    Errores:
        Imprime un mensaje de error si ocurre algún problema durante el despegue.
    """
    try:
        position = get_pose(scf)
        current_z = position[2]  

        if current_z > 0.1:
            print(f"The Crazyflie was already in the air.")
            return 0

        commander = HighLevelCommander(scf.cf)
        commander.takeoff(absolute_height_m=height, duration_s=duration)
        time.sleep(duration)
        print(f"Takeoff completed successfully")

    except Exception as e:
        print(f"ERROR: An error occurred during takeoff: {str(e)}")

def land(scf, height = 0.0, duration = 2.0):
    """
    Comanda al Crazyflie a aterrizar a una altura específica (por defecto al suelo).

    Parámetros:
        scf (SyncCrazyflie): Objeto SyncCrazyflie con la conexión establecida.
        height (float): Altura a la que aterrizar.
        duration (float): Duración del aterrizaje en segundos.

    Errores:
        Imprime un mensaje de error si ocurre algún problema durante el aterrizaje.
    """
    try:
        position = get_pose(scf)
        current_z = position[2]  

        if current_z <= 0.1:
            print(f"The Crazyflie was already on the ground.")
            return 0

        commander = HighLevelCommander(scf.cf)
        commander.land(absolute_height_m=height, duration_s=duration)
        time.sleep(duration)
        commander.stop()
        print(f"Landing completed successfully.")

    except Exception as e:
        print(f"ERROR: An error occurred during landing: {str(e)}")

def move_to_position(scf, x, y, z, velocity = 1.0):
    """
    Comanda al Crazyflie a moverse a una posición específica en el espacio con una velocidad especificada.

    Parámetros:
        scf (SyncCrazyflie): Objeto SyncCrazyflie con la conexión establecida.
        x (float): Coordenada X destino.
        y (float): Coordenada Y destino.
        z (float): Coordenada Z destino.
        velocity (float): Velocidad de movimiento en m/s.

    Errores:
        Imprime un mensaje de error si ocurre algún problema durante el movimiento.
    """
    try:
        commander = scf.cf.high_level_commander
        current_position = get_pose(scf)
        current_x, current_y, current_z = current_position[0], current_position[1], current_position[2]
        distance = ((x - current_x)**2 + (y - current_y)**2 + (z - current_z)**2)**0.5
        duration = distance / velocity
        commander.go_to(x, y, z, yaw=0.0, duration_s=duration)
        time.sleep(duration)
        print(f"Position command completed successfully")

    except Exception as e:
        print(f"ERROR: An error occurred during moving to position: {str(e)}")
