# Sistema automatizado para monitoreo fenológico y de polinizadores

Repositorio asociado al trabajo de tesis:

**“Sistema automatizado para monitoreo fenológico y de polinizadores en campo integrando visión artificial y medición ambiental”.**

Universidad Veracruzana  
Facultad de Ingeniería Eléctrica y Electrónica  
Ingeniería Mecatrónica

## Descripción

Este repositorio contiene los archivos principales utilizados para la implementación del sistema automatizado de monitoreo desarrollado sobre una Raspberry Pi 4. El sistema integra adquisición de imágenes, procesamiento mediante visión artificial, medición de variables ambientales, obtención de coordenadas geográficas, almacenamiento local y consulta de la información.

El subsistema de visión artificial utiliza un modelo basado en YOLO11s como mecanismo principal para la identificación de las clases de interés, mientras que el algoritmo MOG2 se mantiene como mecanismo complementario para la detección de movimiento. El modelo fue posteriormente convertido al formato ONNX y cuantizado a INT8 para su ejecución local en la Raspberry Pi 4.

## Estructura del repositorio

- `codigo/`: contiene los programas principales del sistema, incluyendo la adquisición y procesamiento de imágenes y la lectura de variables ambientales y geográficas.
- `configuracion/`: contiene el archivo `config.yaml`, utilizado para definir los principales parámetros de operación del subsistema visual.
- `panel/`: contiene la interfaz web local utilizada para consultar la información generada por el sistema.
- `services/`: contiene los archivos de servicio y temporización de systemd empleados para automatizar la ejecución de los programas.
- `scripts_red/`: contiene los scripts utilizados para cambiar entre el modo punto de acceso y el modo de conexión a una red Wi-Fi externa.
- `modelos/`: contiene el modelo original `best.pt` y la versión convertida y cuantizada `best_int8.onnx` utilizada en la Raspberry Pi 4.
- `entrenamiento/`: contiene archivos de configuración y resultados generados durante el entrenamiento del modelo YOLO11s.
