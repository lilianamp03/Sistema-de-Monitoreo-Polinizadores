import cv2, time, os, threading, yaml, csv
import numpy as np
import onnxruntime as ort
from datetime import datetime, timedelta

def cargar_config(ruta_config="/home/pi/Desktop/Monitoreo/config.yaml"):
    with open(ruta_config, "r") as f:
        return yaml.safe_load(f)

def silenciar_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)

config = cargar_config()

## Se definen los indices de las camaras
CAM_FOTOS = config["Camaras"]["cam_fotos"]
CAM_DETECCION = config["Camaras"]["cam_eventos"]

## Rutas de guardado de fotos y videos
RUTA_FOTOS = config["Rutas"]["fotos"]
RUTA_EVENTOS = config["Rutas"]["eventos"]
RUTA_REGISTRO = os.path.join(RUTA_EVENTOS, "registro_eventos.csv")

## PARAMETROS CAMARA
INTERVALO_FOTO_MIN = (config["Foto_Periodica"]["intervalo_min"])
FPS_CAM = config["Camara"]["fps_cam"]
FPS_VIDEO = config["Camara"]["fps_video"]
ANCHO_CAM = config["Camara"]["ancho"]
ALTO_CAM = config["Camara"]["alto"]
FORMATO_CAM = config["Camara"]["formato"]

## PARAMETROS MOG2
HISTORY = config["Mog2"]["history"]           		## Numero de frames para construir el fondo, valores grandes -> fondo mas estable pero mas lento de adaptar
VAR_THRESHOLD = config["Mog2"]["var_threshold"]     ## Controla la sensibilidad del detector del movimiento, valores altos -> menor deteccion (mas estricto)
DETECT_SHADOWS = config["Mog2"]["detect_shadows"]  	## Deteccion de sombras des activida

## UMBRAL BINARIO Y FILTRADO DE AREAS
THRESH_BINARY = config["Filtro_movimiento"]["thresh_binary"]     ## Define que valores se definen como objetos o fondo, mascara de 0 a 255, valores arriba de 30 se condidera movimiento
AREA_MIN = config["Filtro_movimiento"]["area_min"]          	 ## Filtrado de ruido y areas grandes
AREA_MAX = config["Filtro_movimiento"]["area_max"]        		 ## Filtrado de ruido y areas grandes

## NUMERO DE ITERACIONES PARA OPERACIONES MORFOLOGICAS
OPEN_ITER = config["Filtro_movimiento"]["open_iter"]           ## ELimina el ruido (puntos blancos)
CLOSE_ITER = config["Filtro_movimiento"]["close_iter"]         ## Une regiones cercanas, rellena huecos

## GUARDADO DEL EVENTO Y DURACION DEL VIDEO
COOLDOWN = config["Eventos"]["cooldown"]
DURACION_VIDEO = (config["Eventos"]["duracion_video"])

SILENCIAR_STDERR = (config["Sistema"]["silenciar_stderr"])

## PREVIEW
MOSTRAR_PREVIEW = (config["Preview"]["mostrar_eventos"])
NOMBRE_VENTANA = (config["Preview"]["nombre_ventana"])

## RED NEURONAL
RED_ACTIVAR = (config["Red_Neuronal"]["activar"])
RUTA_MODELO_YOLO = (config["Red_Neuronal"]["modelo"])
IMG_SIZE = (config["Red_Neuronal"]["imgsz"])
CONF_YOLO = (config["Red_Neuronal"]["confianza"])
USAR_MOG2_RESPALDO = (config["Red_Neuronal"]["usar_mog2_respaldo"])
ESPERA_MOG2_S = (config["Red_Neuronal"].get("espera_mog2_s", 2.0))      
REARME_MOG2_S = (config["Red_Neuronal"].get("rearme_mog2_s", 1.0))

## CLASES DEL MODELO
CLASES = {
    0: "abeja",
    1: "abejorro",
    2: "colibri",
    3: "escarabajo",
    4: "flor",
    5: "hormiga",
    6: "mariposa",
    7: "mosca",
    8: "mosquito",
    9: "polilla"
}

## Flor puede detectarse y mostrarse, pero NO genera evento.
CLASES_EVENTO = {0, 1, 2, 3, 5, 6, 7, 8, 9}

## VARIABLES COMPARTIDAS ENTRE CAMARA Y YOLO
lock_yolo = threading.Lock()
ultimo_frame_yolo = None
nuevo_frame_yolo = threading.Event()
detecciones_yolo = []
evento_yolo_pendiente = False
frame_evento_yolo = None
clases_evento_yolo = []
detalles_evento_yolo = []
tiempo_yolo_ms = 0.0
fps_yolo = 0.0

## FUNCIONES GENERALES

def crear_carpetas():
    os.makedirs(RUTA_FOTOS, exist_ok=True)
    os.makedirs(RUTA_EVENTOS, exist_ok=True)

def timestamp_nombre():
    return datetime.now().strftime("%d-%m-%Y_%H%M%S")

def registrar_evento(fecha, origen, detalles, imagen, video):
    nuevo_archivo = not os.path.exists(RUTA_REGISTRO) or os.path.getsize(RUTA_REGISTRO) == 0

    if origen == "YOLO" and detalles:
        especies = "; ".join(nombre for nombre, _ in detalles)
        confianzas = "; ".join(f"{conf:.2f}" for _, conf in detalles)
    else:
        especies = "no_clasificada"
        confianzas = ""

    with open(RUTA_REGISTRO, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if nuevo_archivo:
            writer.writerow(["fecha", "hora", "origen", "especies", "confianzas", "imagen", "video"])

        writer.writerow([
            fecha.strftime("%d/%m/%Y"),
            fecha.strftime("%H:%M:%S"),
            origen,
            especies,
            confianzas,
            os.path.basename(imagen),
            os.path.basename(video) if video else ""
        ])

def abrir_cam(index, fps):
    cam = cv2.VideoCapture(index, cv2.CAP_V4L2)

    if not cam.isOpened():
        print(f"Error. No se pudo abrir la camara {index}", flush=True)
        return None
    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAM)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAM)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FORMATO_CAM))
    cam.set(cv2.CAP_PROP_FPS, fps)
    ## Buffer para evitar frames atrasados
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ## Pequenio calentamiento
    time.sleep(1)
    
    for _ in range(10):
        cam.read()
        time.sleep(0.03)

    print(f"Camara {index} abierta",flush=True)
    return cam

## MODO 1: FOTO PERIODICA
def tomar_foto(cam):
    ahora = datetime.now()
    timestamp = timestamp_nombre()
    nombre = f"foto_{timestamp}.jpg"
    ruta = os.path.join(RUTA_FOTOS, nombre)

    image = None

    ## Calentamiento de 1 segundo para estabilizar la imagen
    t0 = time.time()
    while time.time() - t0 < 1.0:
        ret, frame = cam.read()
        if ret:
            image = frame
        time.sleep(0.03)

    if image is None:
        print("No se pudo capturar la imagen", flush=True)
        return False

    sello = ahora.strftime("%d/%m/%Y %H:%M:%S")
    cv2.putText(image, sello, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    ok = cv2.imwrite(ruta, image)

    if ok:
        ruta_panel = "/home/pi/Desktop/Monitoreo/Panel/ultima.jpg"
        ruta_panel_ap = "/home/pi/Desktop/Archivos_public/ultima.jpg"
        cv2.imwrite(ruta_panel, image)
        cv2.imwrite(ruta_panel_ap, image)
        print(f"Imagen guardada: {ruta}", flush=True)
        print(f"Imagen del panel actualizada: {ruta_panel}", flush=True)
    else:
        print(f"Imagen NO guardada: {ruta}", flush=True)
        
    return ok

def modo_foto():
    print("Modo foto periodica iniciado", flush=True)
    cam = abrir_cam(CAM_FOTOS, FPS_CAM)

    if cam is None:
        print("No se pudo iniciar la camara de fotos", flush=True)
        return
    try:
        while True:
            t_next = (datetime.now() + timedelta(minutes=INTERVALO_FOTO_MIN))
            tomar_foto(cam)
            
            while True:
                restante = (t_next - datetime.now()).total_seconds()

                if restante <= 0:
                    break
                    
                time.sleep(min(1.0,restante))

    except KeyboardInterrupt:
        print("Salida limpia del modo foto", flush=True)

    finally:
        cam.release()
        print("Camara de fotos cerrada", flush=True)

## MOG2
def calibrar_mog2(cap, bg, n_frames=30):
    for _ in range(n_frames):
        ret, frame = (cap.read())
        
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        bg.apply(blur)

def procesar_mog2(frame, bg):
    global AREA_MIN, AREA_MAX

    if AREA_MAX < AREA_MIN:
        AREA_MAX = (AREA_MIN + 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    fg_mask = bg.apply(blur)
    
    ## Binarizacion
    _, mask = cv2.threshold(fg_mask, THRESH_BINARY, 255, cv2.THRESH_BINARY)
    ## Kernel eliptico para operaciones morfologicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    if OPEN_ITER > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITER)
    if CLOSE_ITER > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITER)

    contours, _ = (cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))

    cajas = []

    for c in contours:
        area = (cv2.contourArea(c))

        if (AREA_MIN <= area <= AREA_MAX):
            x, y, w, h = (cv2.boundingRect(c))
            cajas.append((x, y, w, h))

    evento = (len(cajas) > 0)
    
    return evento, cajas

def dibujar_mog2(frame, cajas):
    for i, caja in enumerate(cajas, start=1):
        x, y, w, h = (caja)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, f"Movimiento {i}",(x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame

## PROCESAMIENTO YOLO
def letterbox_yolo(frame):
    alto, ancho = (frame.shape[:2])
    escala = min(IMG_SIZE / ancho, IMG_SIZE / alto)
    nuevo_ancho = int(round(ancho * escala))
    nuevo_alto = int(round(alto * escala))

    imagen = cv2.resize(frame,(nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LINEAR)
    
    pad_ancho = (IMG_SIZE - nuevo_ancho)
    pad_alto = (IMG_SIZE - nuevo_alto)
    izquierda = (pad_ancho // 2)
    derecha = (pad_ancho - izquierda)
    arriba = (pad_alto // 2)
    abajo = (pad_alto - arriba)

    imagen = (cv2.copyMakeBorder(imagen, arriba, abajo, izquierda, derecha, cv2.BORDER_CONSTANT, value=(114, 114, 114)))

    return (imagen, escala, izquierda, arriba)

def preparar_yolo(frame):
    (imagen, escala, pad_x, pad_y) = letterbox_yolo(frame)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen = imagen.transpose(2, 0, 1)
    imagen = (imagen.astype(np.float32)/255.0)
    imagen = (np.expand_dims(imagen, axis=0))
    imagen = (np.ascontiguousarray(imagen))

    return (imagen, escala, pad_x, pad_y)

def ajustar_caja_yolo(x1, y1, x2, y2, escala, pad_x, pad_y, ancho_original, alto_original):
    x1 = (x1 - pad_x) / escala
    y1 = (y1 - pad_y) / escala
    x2 = (x2 - pad_x) / escala
    y2 = (y2 - pad_y) / escala
    x1 = int(max(0, min(ancho_original - 1, x1)))
    y1 = int(max(0, min(alto_original - 1, y1)))
    x2 = int(max(0, min(ancho_original - 1, x2)))
    y2 = int(max(0, min(alto_original - 1, y2)))

    return x1, y1, x2, y2

def dibujar_yolo(frame, detecciones):
    for det in detecciones:
        (x1, y1, x2, y2, nombre, conf, clase) = det
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{nombre} {conf:.2f}",(x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    return frame

## HILO YOLO
def ejecutar_yolo(sesion, nombre_entrada, nombre_salida, detener_yolo):
    global ultimo_frame_yolo, detecciones_yolo, evento_yolo_pendiente, frame_evento_yolo, clases_evento_yolo, detalles_evento_yolo, tiempo_yolo_ms, fps_yolo

    while not detener_yolo.is_set():
        if not nuevo_frame_yolo.wait(timeout=0.2):
            continue

        nuevo_frame_yolo.clear()
        with lock_yolo:
            if ultimo_frame_yolo is None:
                continue

            frame = ultimo_frame_yolo.copy()

        alto_original, ancho_original = (frame.shape[:2])
        
        (entrada_modelo, escala, pad_x, pad_y) = preparar_yolo(frame)

        ## INFERENCIA
        inicio = (time.perf_counter())
        resultado = (sesion.run([nombre_salida],{nombre_entrada:entrada_modelo})[0][0])
        fin = (time.perf_counter())
        tiempo = (fin - inicio)

        nuevas_detecciones = []
        hay_polinizador = False
        nombres_polinizadores = []
        detalles_polinizadores = []

        ## RESULTADOS

        for det in resultado:
            (x1, y1, x2, y2, conf, clase) = det
            if conf < CONF_YOLO:
                continue

            clase = int(clase)
            if clase not in CLASES:
                continue

            nombre = (CLASES[clase])
            (x1, y1, x2, y2) = ajustar_caja_yolo(x1, y1, x2, y2, escala, pad_x, pad_y, ancho_original, alto_original)
            nuevas_detecciones.append((x1, y1, x2, y2, nombre, float(conf), clase))

            # FLOR NO entra aqui
            if clase in CLASES_EVENTO:
                hay_polinizador = True
                detalles_polinizadores.append((nombre, float(conf)))
                if (nombre not in nombres_polinizadores): 
                    nombres_polinizadores.append(nombre)

        frame_anotado = (frame.copy())

        dibujar_yolo(frame_anotado, nuevas_detecciones)
        tiempo_ms = (tiempo * 1000.0)

        fps_actual = (1.0 / tiempo if tiempo > 0 else 0.0)

        ## Actualizar resultados
        with lock_yolo:
            detecciones_yolo = (nuevas_detecciones)
            tiempo_yolo_ms = tiempo_ms

            fps_yolo = fps_actual

            if hay_polinizador:
                evento_yolo_pendiente = True
                frame_evento_yolo = frame_anotado.copy()
                clases_evento_yolo = (nombres_polinizadores.copy())
                detalles_evento_yolo = (detalles_polinizadores.copy())

## MODO 2: YOLO+ MOG2 (RESPALDO)
def modo_deteccion():
    global ultimo_frame_yolo, evento_yolo_pendiente, frame_evento_yolo, clases_evento_yolo, detalles_evento_yolo
    print("Modo deteccion iniciado", flush=True)

    ## Abrir camara 
    cap = abrir_cam(CAM_DETECCION, FPS_CAM)
    if cap is None:
        print("No se pudo abrir la camara de deteccion", flush=True)
        return

    ## Inicializar MOG2
    bg = None
    if USAR_MOG2_RESPALDO:
        bg = cv2.createBackgroundSubtractorMOG2(history=HISTORY, varThreshold=VAR_THRESHOLD, detectShadows=DETECT_SHADOWS)
        print("Calibrando fondo MOG2", flush=True)
        calibrar_mog2(cap, bg, n_frames=FPS_CAM)
        print("Fondo MOG2 calibrado", flush=True)

    ## Inicializar YOLO
    sesion_yolo = None
    hilo_yolo = None
    detener_yolo = (threading.Event())

    if RED_ACTIVAR:
        try:
            print("Cargando modelo YOLO ONNX...", flush=True)
            opciones = ort.SessionOptions()
            opciones.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

            sesion_yolo = (ort.InferenceSession(RUTA_MODELO_YOLO, sess_options=opciones, providers=["CPUExecutionProvider"]))

            nombre_entrada = (sesion_yolo.get_inputs()[0].name)
            nombre_salida = (sesion_yolo.get_outputs()[0].name)

            print("Modelo YOLO cargado", flush=True)
            print(f"Modelo: {RUTA_MODELO_YOLO}", flush=True)
            print(f"Resolucion YOLO: {IMG_SIZE}", flush=True)
            print(f"Confianza YOLO: {CONF_YOLO}", flush=True)

            ## Calentamiento del modelo
            dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32) 
            print("Calentando modelo YOLO...", flush=True)

            for _ in range(3):
                sesion_yolo.run([nombre_salida], {nombre_entrada: dummy})

            print("YOLO listo", flush=True)

            hilo_yolo = (threading.Thread(target=ejecutar_yolo, args=(sesion_yolo, nombre_entrada, nombre_salida, detener_yolo), daemon=True))
            hilo_yolo.start()

        except Exception as e:
            print("ERROR cargando YOLO:",e, flush=True)
            print("El sistema continuara solo con MOG2", flush=True)
            sesion_yolo = None

    ## Control de eventos
    ultimo_guardado = 0.0

    ## Control del respaldo MOG2
    inicio_movimiento_mog2 = None
    ultimo_movimiento_mog2 = 0.0
    mog2_bloqueado = False

    ## Video
    fourcc = (cv2.VideoWriter_fourcc(*"MJPG"))
    vw = None
    grabando = False
    fin_video = 0.0
    origen_video = ""
    ruta_video_actual = ""

    ## Ciclo principal
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el frame de deteccion", flush=True) 
                time.sleep(0.2)
                continue

            frame_original = frame.copy()

            ## Entregar Frame reciente a YOLO
            if (RED_ACTIVAR and sesion_yolo is not None):
                with lock_yolo:
                    ultimo_frame_yolo = frame_original.copy()
                nuevo_frame_yolo.set()

            ## Procesar MOG2
            evento_mog2 = False
            cajas_mog2 = []
            if USAR_MOG2_RESPALDO and bg is not None:
                (evento_mog2, cajas_mog2) = procesar_mog2(frame_original, bg)


            ## Leer resultado mas reciente de YOLO
            evento_yolo = False
            frame_yolo_evento = None
            nombres_yolo_evento = []
            detalles_yolo_evento = []
            det_actuales = []
            yolo_ms_actual = 0.0
            yolo_fps_actual = 0.0

            if (RED_ACTIVAR and sesion_yolo is not None):
                with lock_yolo:
                    det_actuales = list(detecciones_yolo)
                    yolo_ms_actual = tiempo_yolo_ms
                    yolo_fps_actual = fps_yolo

                    if evento_yolo_pendiente:
                        evento_yolo = True
                        if (frame_evento_yolo is not None):
                            frame_yolo_evento = frame_evento_yolo.copy()
                        
                        nombres_yolo_evento = list(clases_evento_yolo)
                        detalles_yolo_evento = list(detalles_evento_yolo)
                        ## Consumir evento pendiente.
                        evento_yolo_pendiente = False
                        frame_evento_yolo = None
                        clases_evento_yolo = []
                        detalles_evento_yolo = []

            # Saber si YOLO sigue viendo un polinizador
            yolo_detectando_polinizador = any(det[6] in CLASES_EVENTO for det in det_actuales)

            ## Frame visual
            frame_visual = frame_original.copy()
            # Dibujar ultima deteccion disponible de YOLO
            dibujar_yolo(frame_visual, det_actuales)

            ## Control de respaldo de MOG2
            ahora = time.time()
            ## MOG2 ve movimiento
            if evento_mog2:
                ultimo_movimiento_mog2 = ahora
                if (inicio_movimiento_mog2 is None):
                    inicio_movimiento_mog2 = ahora

            ## Escena sin movimiento
            else:
                if (ultimo_movimiento_mog2 > 0 and (ahora - ultimo_movimiento_mog2 ) >= REARME_MOG2_S):
                    inicio_movimiento_mog2 = None
                    mog2_bloqueado = False

            ## Mostrar cajas solo cuando MOG2 actua como respaldo
            if (USAR_MOG2_RESPALDO and evento_mog2 and not yolo_detectando_polinizador and not evento_yolo):
                dibujar_mog2(frame_visual, cajas_mog2)

            ## Decision del evento
            evento = False
            origen = ""
            frame_evento = None

            ## 1. Deteccion de evento con YOLO
            if evento_yolo:
                evento = True
                origen = "YOLO"
                frame_evento = (frame_yolo_evento if frame_yolo_evento is not None else frame_visual.copy())

                # Este episodio de movimiento ya fue atendido por YOLO.
                mog2_bloqueado = True

            # 2. MOG2 actua como respaldo
            elif (USAR_MOG2_RESPALDO and evento_mog2 and not yolo_detectando_polinizador and not mog2_bloqueado and inicio_movimiento_mog2 is not None):
                tiempo_movimiento = (ahora - inicio_movimiento_mog2)
                
                ## Dar tiempo a YOLO antes de disparar el respaldo.
                if (tiempo_movimiento >= ESPERA_MOG2_S):
                    evento = True
                    origen = "MOG2"
                    frame_evento = frame_original.copy()

                    dibujar_mog2(frame_evento, cajas_mog2)
                    # Solo un evento MOG2 durante el mismo episodio continuo de movimiento.
                    mog2_bloqueado = True

            registro_pendiente = None

            ## Guardar imagen
            if (evento and (ahora - ultimo_guardado) >= COOLDOWN):
                fecha_evento = datetime.now()
                timestamp = fecha_evento.strftime("%d-%m-%Y_%H%M%S")
                nombre_img = os.path.join(RUTA_EVENTOS,(f"evento_img_{timestamp}_{origen.lower()}.jpg"))

                ok = cv2.imwrite(nombre_img, frame_evento)
                ultimo_guardado = ahora

                if ok:
                    print("\nEVENTO DETECTADO", flush=True)
                    print(f"Origen: {origen}",flush=True)

                    if (origen == "YOLO" and nombres_yolo_evento):
                        print("Clases:",", ".join(nombres_yolo_evento), flush=True)
                    print(f"Imagen: {nombre_img}", flush=True)

                    registro_pendiente = (
                        fecha_evento,
                        origen,
                        detalles_yolo_evento.copy() if origen == "YOLO" else [],
                        nombre_img
                    )

            ## Iniciar video
            if (evento and not grabando):
                timestamp = timestamp_nombre()
                ruta_video = (os.path.join(RUTA_EVENTOS,(f"evento_vid_{timestamp}_{origen.lower()}.avi")))
                alto, ancho = (frame_visual.shape[:2])
                vw = cv2.VideoWriter(ruta_video, fourcc, FPS_VIDEO,(ancho, alto))

                if vw.isOpened():
                    grabando = True
                    origen_video = origen
                    ruta_video_actual = ruta_video
                    fin_video = (time.time() + DURACION_VIDEO)
                    print(f"Grabando video de evento {origen}: {ruta_video}", flush=True)
                else:
                    vw = None
                    grabando = False
                    ruta_video_actual = ""
                    print("No se pudo iniciar VideoWriter", flush=True)

            if registro_pendiente is not None:
                fecha_registro, origen_registro, detalles_registro, imagen_registro = registro_pendiente
                registrar_evento(
                    fecha_registro,
                    origen_registro,
                    detalles_registro,
                    imagen_registro,
                    ruta_video_actual if grabando else ""
                )

            ## Grabar video
            if (grabando and vw is not None):
                vw.write(frame_visual)
                if (time.time()>=fin_video):
                    vw.release()
                    vw = None
                    grabando = False
                    ruta_video_actual = ""
                    print(f"Video de evento {origen_video} terminado", flush=True)

            ## Preview
            if MOSTRAR_PREVIEW:
                cv2.putText(frame_visual, (f"YOLO: {yolo_fps_actual:.2f} FPS"), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(frame_visual, (f"Inferencia: {yolo_ms_actual:.0f} ms"), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(frame_visual, ("YOLO principal | MOG2 respaldo"), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

                if yolo_detectando_polinizador:
                    cv2.putText(frame_visual,"Estado: YOLO detectando",(10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
                elif mog2_bloqueado:
                    cv2.putText(frame_visual,"Estado: MOG2 bloqueado",(10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
                elif evento_mog2:
                    cv2.putText(frame_visual,"Estado: MOG2 esperando YOLO", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50,(0, 255, 255), 2)
                else:
                    cv2.putText(frame_visual,"Estado: monitoreando",(10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

                cv2.imshow(NOMBRE_VENTANA, frame_visual )

                if (cv2.waitKey(1) & 0xFF == ord("q")):
                    print("Preview cerrado por usuario", flush=True)
                    break

            time.sleep(0.001)

    except KeyboardInterrupt:
        print(
            "Salida limpia del modo deteccion", flush=True)

    finally:
        # Detener hilo YOLO.
        detener_yolo.set()
        nuevo_frame_yolo.set()

        if hilo_yolo is not None:
            hilo_yolo.join(timeout=3)

        if vw is not None:
            vw.release()

        if MOSTRAR_PREVIEW:
            cv2.destroyAllWindows()

        cap.release()

        print("Camara de deteccion cerrada", flush=True)

## PROGRAMA PRINCIPAL
def main():
    if SILENCIAR_STDERR:
        silenciar_stderr()
    print("-------------------------------------", flush=True)
    print("SISTEMA DE MONITOREO", flush=True)
    print("YOLO INT8 + MOG2", flush=True)
    print("-------------------------------------", flush=True)
    print(f"Camara para fotos: {CAM_FOTOS}", flush=True)
    print(f"Camara de deteccion: {CAM_DETECCION}", flush=True)
    print(f"Intervalo de foto: {INTERVALO_FOTO_MIN} min", flush=True)
    print(f"Resolucion de camara: {ANCHO_CAM}x{ALTO_CAM}", flush=True)
    print(f"FPS camara: {FPS_CAM}", flush=True)
    print(f"Modelo YOLO: {RUTA_MODELO_YOLO}",flush=True)
    print(f"Resolucion YOLO: {IMG_SIZE}x{IMG_SIZE}", flush=True)
    print(f"Confianza YOLO: {CONF_YOLO}", flush=True)
    print(f"YOLO activado: {RED_ACTIVAR}", flush=True)
    print(f"MOG2 respaldo: {USAR_MOG2_RESPALDO}", flush=True)
    print(f"Espera MOG2: {ESPERA_MOG2_S} s", flush=True)
    print(f"Rearme MOG2: {REARME_MOG2_S} s", flush=True)

    crear_carpetas()

    ## HILO CON LOS 2 SUBSISTEMAS PRINCIPALES
    hilo_fotos = (threading.Thread(target=modo_foto))
    hilo_deteccion = (threading.Thread(target=modo_deteccion))
    hilo_fotos.start()
    hilo_deteccion.start()
    hilo_fotos.join()
    hilo_deteccion.join()

if __name__ == "__main__":
    main()