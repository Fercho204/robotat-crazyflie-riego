# Planificación de rutas para riego con Crazyflie en el Robotat

Código del trabajo de graduación "Implementación de un sistema a escala de planificación de rutas para riego de cultivos con drones dentro del ecosistema Robotat", de la Universidad del Valle de Guatemala.

El programa deja dibujar un área de cultivo, genera una ruta de cobertura en zig-zag, la simula y la vuela con un Crazyflie 2.1. La posición del dron la mide OptiTrack y llega por MQTT. Con la cámara del AI-deck y un modelo de profundidad, el dron detecta obstáculos en la ruta y los rodea. El firmware del Crazyflie y del AI-deck es el oficial de Bitcraze, sin cambios.

## Requisitos

- Windows 10 u 11. Los scripts leen el teclado con `msvcrt`, que solo existe en Windows.
- Python 3.11.
- Crazyflie 2.1 con AI-deck 1.1, Crazyradio y marcadores de OptiTrack.
- Acceso a la red del Robotat (broker MQTT `192.168.50.200`, puerto `1880`, tópico `mocap/drone1`) y a la red WiFi del AI-deck (`192.168.4.1`).

## Instalación

```powershell
py -3.11 -m pip install -r requirements.txt
# Una vez, con internet, para descargar los modelos de profundidad
py -3.11 -c "import Prueba_ruta_vision as p; p.precargar_midas('hibrido')"
```

El Crazyradio necesita el controlador libusb, que se instala con Zadig siguiendo la guía de Bitcraze.

## Uso

```powershell
py -3.11 Prueba_despegue.py        # prueba de radio y motores, sin OptiTrack
py -3.11 Prueba_ruta_carrot.py     # planificador y vuelo de la ruta
py -3.11 Prueba_ruta_carrot.py --calibrar        # montaje de la cámara, no vuela
py -3.11 Prueba_ruta_carrot.py --calibrar-midas  # distancia del modelo de profundidad, no vuela
```

Hay que lanzarlos desde esta carpeta. En el planificador se dibuja el área con clics y se cierra con Enter, se revisa la simulación, se puede guardar la ruta en `rutas_guardadas` y se confirma con VOLAR. Después se elige el cultivo, que fija la altura de vuelo.

**Paro de emergencia:** la tecla `q` en la consola, o `q` y `ESC` en la ventana de la cámara, apaga los motores al instante.

## Archivos

| Archivo | Qué hace |
|---|---|
| `Prueba_ruta_carrot.py` | Programa principal. Vuelo con zanahoria móvil y evasión de obstáculos |
| `Prueba_ruta_vision.py` | Librería con la interfaz, el MoCap por MQTT, la cámara, el paro de emergencia y la generación de rutas |
| `crazyflie_python_commands_mod.py` | Conexión con el Crazyflie y reinicio del filtro de Kalman |
| `evasion.py`, `maniobra_evasion.py` | Detección de obstáculos y maniobra para rodearlos |
| `pursuit_ruta.py`, `trayectoria_suave.py` | Geometría y perfil de velocidad de la ruta |
| `puente_camara.py` | Recibe las imágenes del AI-deck en un proceso aparte |
| `Prueba_despegue.py` | Despegue y aterrizaje en lazo abierto para probar la radio |
| `probar_*_offline.py`, `banco_headless.py` | Pruebas que corren sin dron, cámara ni OptiTrack |
| `pruebas_camara/ultimos_ajustes.json` | Ajustes de imagen que se cargan al arrancar |

El manual de uso completo, con capturas, la tabla de parámetros y los problemas comunes, está en el repositorio de la universidad, en `documentos/manuales/manual_de_uso.docx`.
