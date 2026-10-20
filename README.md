# Sistema automatizado para monitoreo fenológico y de polinizadores

Repositorio asociado al trabajo de tesis:

**“Sistema automatizado para monitoreo fenológico y de polinizadores en campo integrando visión artificial y medición ambiental”.**

**Universidad Veracruzana**  
Facultad de Ingeniería Eléctrica y Electrónica  
Ingeniería Mecatrónica  

## Descripción

Este repositorio contiene los principales archivos utilizados para el desarrollo e implementación de un sistema automatizado de monitoreo basado en una Raspberry Pi 4.

El sistema integra la adquisición de información visual, ambiental y geográfica, así como el procesamiento de imágenes mediante visión artificial, almacenamiento local y consulta de los registros generados.

Para la adquisición visual se utilizaron dos cámaras USB con funciones independientes. Una de ellas realiza capturas periódicas para generar un registro visual de la vegetación, mientras que la segunda se utiliza para el monitoreo mediante visión artificial.

El sistema utiliza un modelo basado en YOLO11s para identificar las clases de interés y mantiene el algoritmo MOG2 como un mecanismo complementario para la detección de movimiento. El modelo entrenado fue posteriormente convertido al formato ONNX y cuantizado a INT8 para su ejecución local en la Raspberry Pi 4.

También se incorporó un sensor BME688 para adquirir variables ambientales y un módulo GPS GY-GPS6MV2 para obtener las coordenadas geográficas correspondientes al sitio de monitoreo.

## Componentes principales

El sistema fue implementado utilizando:

- Raspberry Pi 4.
- Dos cámaras USB.
- Sensor ambiental BME688.
- Módulo GPS GY-GPS6MV2.
- Memoria microSD para almacenamiento local.
- Modelo de detección basado en YOLO11s.
- Algoritmo MOG2 para detección de movimiento.
- Python para el desarrollo de los programas principales.
- OpenCV para adquisición y procesamiento de imágenes.
- ONNX Runtime para la ejecución del modelo cuantizado.
- Interfaz web local para consulta de los registros.

## Funcionamiento general

El sistema se divide principalmente en dos rutas de adquisición visual.

La primera cámara realiza capturas periódicas de la zona observada. En la configuración utilizada durante el desarrollo se estableció una resolución de 640 × 480 píxeles y un intervalo de captura de 30 minutos.

La segunda cámara mantiene una adquisición continua de imágenes para el subsistema de visión artificial. El modelo YOLO11s se utiliza como método principal para identificar las clases de interés, mientras que MOG2 puede utilizarse como mecanismo complementario basado en movimiento.

De forma independiente, el programa encargado del BME688 y del GPS obtiene registros de temperatura, humedad relativa, presión atmosférica, resistencia del elemento sensible a gases, latitud y longitud. Estos datos se almacenan junto con una referencia temporal.

Los diferentes procesos se automatizan mediante servicios y temporizadores de systemd.

## Clases del modelo

El modelo de detección fue entrenado considerando diez clases:

- abeja
- abejorro
- colibrí
- escarabajo
- flor
- hormiga
- mariposa
- mosca
- mosquito
- polilla

Estas clases corresponden a las categorías utilizadas durante la preparación y etiquetado del conjunto de datos.

## Modelo de detección

Para el entrenamiento se utilizó YOLO11s mediante la biblioteca Ultralytics.

La configuración principal utilizada fue:

- Modelo base: `yolo11s.pt`
- Número máximo de épocas: 100
- Resolución de entrada: 640 × 640 píxeles
- Tamaño de lote: 16
- Paciencia: 20 épocas
- Optimizador: AdamW

Como resultado del entrenamiento se obtuvo el archivo `best.pt`.

Posteriormente, el modelo fue convertido al formato ONNX y cuantizado a INT8 para su implementación en la Raspberry Pi 4. La versión utilizada en el sistema corresponde al archivo `best_int8.onnx`.

## Resultados principales

Sobre el conjunto independiente de prueba, el modelo original obtuvo:

- Precisión: 0.9161
- Recall: 0.8945
- mAP@0.5: 0.9195
- mAP@0.5:0.95: 0.5881

La versión cuantizada `best_int8.onnx` pudo ejecutarse directamente en la Raspberry Pi 4. Durante las pruebas se obtuvo un tiempo promedio de inferencia de 789.71 ms, equivalente aproximadamente a 1.27 inferencias por segundo.

## Estructura del repositorio

```text
Sistema-Monitoreo-RPi4/
│
├── codigo/
│   ├── Monitoreo_YOLO.py
│   └── Bme_Gps.py
│
├── configuracion/
│   └── config.yaml
│
├── panel/
│   └── index.html
│
├── services/
│   ├── monitoreo.service
│   ├── bme_gps.service
│   └── bme_gps.timer
│
├── scripts_red/
│   ├── wifi-ap.txt
│   └── wifi-internet.txt
│
├── modelos/
│   ├── best.pt
│   └── best_int8.onnx
│
├── entrenamiento/
│   ├── args.yaml
│   ├── results.csv
│   ├── results.png
│   ├── BoxPR_curve.png
│   └── confusion_matrix_normalized.png
│
└── README.md
