# Sistema automatizado para monitoreo fenológico y de polinizadores

Repositorio asociado al trabajo de tesis:

**“Sistema automatizado para monitoreo fenológico y de polinizadores en campo integrando visión artificial y medición ambiental”.**

Universidad Veracruzana  
Facultad de Ingeniería Eléctrica y Electrónica  
Ingeniería Mecatrónica

## Descripción

Este repositorio contiene los archivos principales utilizados para la implementación del sistema automatizado de monitoreo desarrollado sobre una Raspberry Pi 4. El sistema integra adquisición de imágenes, procesamiento mediante visión artificial, medición de variables ambientales, obtención de coordenadas geográficas, almacenamiento local y consulta de la información.

El subsistema de visión artificial utiliza un modelo basado en YOLO11s como mecanismo principal para la identificación de las clases de interés, mientras que el algoritmo MOG2 se mantiene como mecanismo complementario para la detección de movimiento. El modelo fue posteriormente convertido al formato ONNX y cuantizado a INT8 para su ejecución local en la Raspberry Pi 4.
