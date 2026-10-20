import cv2, time, os, threading, yaml, csv
import numpy as np
import onnxruntime as ort
from datetime import datetime

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

## Copias de la ultima foto (panel y carpeta publica), opcionales en el YAML
RUTA_PANEL = config["Rutas"].get("panel", "/home/pi/Desktop/Monitoreo/Panel")
RUTA_PUBLICA = config["Rutas"].get("publica", "/home/pi/Desktop/Archivos_public")

## PARAMETROS CAMARA
INTERVALO_FOTO_MIN = config["Foto_Periodica"]["intervalo_min"]
FPS_CAM = config["Camara"]["fps_cam"]
FPS_VIDEO = config["Camara"]["fps_video"]
ANCHO_CAM = config["Camara"]["ancho"]
ALTO_CAM = config["Camara"]["alto"]
FORMATO_CAM = config["Camara"]["formato"]

## PARAMETROS MOG2
HISTORY = config["Mog2"]["history"]           		## Numero de frames para construir el fondo, valores grandes -> fondo mas estable pero mas lento de adaptar
VAR_THRESHOLD = config["Mog2"]["var_threshold"]     ## Controla la sensibilidad del detector del movimiento, valores altos -> menor deteccion (mas estricto)
DETECT_SHADOWS = config["Mog2"]["detect_shadows"]  	## Deteccion de sombras desactivada

## UMBRAL BINARIO Y FILTRADO DE AREAS
THRESH_BINARY = config["Filtro_movimiento"]["thresh_binary"]     ## Define que valores se consideran objetos o fondo, mascara de 0 a 255
AREA_MIN = config["Filtro_movimiento"]["area_min"]          	 ## Filtrado de ruido (areas pequenias)
AREA_MAX = max(config["Filtro_movimiento"]["area_max"], AREA_MIN + 1)  ## Filtrado de areas grandes, siempre mayor que AREA_MIN

## NUMERO DE ITERACIONES PARA OPERACIONES MORFOLOGICAS
OPEN_ITER = config["Filtro_movimiento"]["open_iter"]           ## Elimina el ruido (puntos blancos)
CLOSE_ITER = config["Filtro_movimiento"]["close_iter"]         ## Une regiones cercanas, rellena huecos

## GUARDADO DEL EVENTO Y DURACION DEL VIDEO
COOLDOWN = config["Eventos"]["cooldown"]
DURACION_VIDEO = config["Eventos"]["duracion_video"]

SILENCIAR_STDERR = config["Sistema"]["silenciar_stderr"]

## PREVIEW
MOSTRAR_PREVIEW = config["Preview"]["mostrar_eventos"]
NOMBRE_VENTANA = config["Preview"]["nombre_ventana"]

## RED NEURONAL
RED_ACTIVAR = config["Red_Neuronal"]["activar"]
RUTA_MODELO_YOLO = config["Red_Neuronal"]["modelo"]
IMG_SIZE = config["Red_Neuronal"]["imgsz"]
CONF_YOLO = config["Red_Neuronal"]["confianza"]
USAR_MOG2_RESPALDO = config["Red_Neuronal"]["usar_mog2_respaldo"]
ESPERA_MOG2_S = config["Red_Neuronal"].get("espera_mog2_s", 2.0)      ## Tiempo que se le da a YOLO antes de que dispare MOG2
REARME_MOG2_S = config["Red_Neuronal"].get("rearme_mog2_s", 1.0)      ## Tiempo de quietud para rearmar MOG2

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

## Kernel eliptico para operaciones morfologicas (se crea una sola vez)
KERNEL_MORF = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

## Senial de paro (Ctrl+C) compartida por los hilos
detener = threading.Event()

## FUNCIONES GENERALES

def crear_carpetas():
    for ruta in (RUTA_FOTOS, RUTA_EVENTOS, RUTA_PANEL, RUTA_PUBLICA):
        os.makedirs(ruta, exist_ok=True)

def timestamp_nombre():
    return datetime.now().strftime("%d-%m-%Y_%H%M%S")

def registrar_evento(fecha, origen, detalles, imagen, video):
    ## Agrega una fila al CSV, si el archivo es nuevo escribe primero el encabezado
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

    print(f"Camara {index} abierta", flush=True)
    return cam

## MODO 1: FOTO PERIODICA
def tomar_foto(cam):
    ahora = datetime.now()
    ruta = os.path.join(RUTA_FOTOS, f"foto_{timestamp_nombre()}.jpg")

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

    if not cv2.imwrite(ruta, image):
        print(f"Imagen NO guardada: {ruta}", flush=True)
        return False

    print(f"Imagen guardada: {ruta}", flush=True)

    ## Ultima foto para el panel y la carpeta publica
    for carpeta in (RUTA_PANEL, RUTA_PUBLICA):
        cv2.imwrite(os.path.join(carpeta, "ultima.jpg"), image)

    return True

def modo_foto():
    print("Modo foto periodica iniciado", flush=True)
    cam = abrir_cam(CAM_FOTOS, FPS_CAM)

    if cam is None:
        print("No se pudo iniciar la camara de fotos", flush=True)
        return

    try:
        while not detener.is_set():
            siguiente = time.time() + INTERVALO_FOTO_MIN * 60
            tomar_foto(cam)

            ## Espera hasta la siguiente foto, se interrumpe si hay paro
            if detener.wait(timeout=max(0.0, siguiente - time.time())):
                break

    finally:
        cam.release()
        print("Camara de fotos cerrada", flush=True)

## MOG2
def dibujar_mog2(frame, cajas):
    for i, (x, y, w, h) in enumerate(cajas, start=1):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, f"Movimiento {i}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame

## Detector de movimiento + control de cuando puede disparar un evento
class RespaldoMog2:
    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=HISTORY, varThreshold=VAR_THRESHOLD, detectShadows=DETECT_SHADOWS)
        self.inicio = None        ## Cuando empezo el movimiento actual
        self.ultimo = 0.0         ## Ultima vez que se vio movimiento
        self.bloqueado = False    ## True si este episodio de movimiento ya genero evento

    def _preparar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (21, 21), 0)

    def calibrar(self, cap, n_frames):
        for _ in range(n_frames):
            ret, frame = cap.read()

            if not ret:
                break

            self.bg.apply(self._preparar(frame))

    def procesar(self, frame):
        ## Devuelve la lista de cajas (x, y, w, h) con movimiento valido
        mascara = self.bg.apply(self._preparar(frame))

        ## Binarizacion
        _, mascara = cv2.threshold(mascara, THRESH_BINARY, 255, cv2.THRESH_BINARY)

        if OPEN_ITER > 0:
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, KERNEL_MORF, iterations=OPEN_ITER)
        if CLOSE_ITER > 0:
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, KERNEL_MORF, iterations=CLOSE_ITER)

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return [cv2.boundingRect(c) for c in contornos if AREA_MIN <= cv2.contourArea(c) <= AREA_MAX]

    def actualizar(self, hay_movimiento, ahora):
        ## Lleva el control del episodio de movimiento y lo rearma cuando la escena se calma
        if hay_movimiento:
            self.ultimo = ahora
            if self.inicio is None:
                self.inicio = ahora

        elif self.ultimo > 0 and (ahora - self.ultimo) >= REARME_MOG2_S:
            self.inicio = None
            self.bloqueado = False

    def debe_disparar(self, hay_movimiento, yolo_ve, ahora):
        ## True si hay movimiento sostenido que YOLO no logro identificar
        if not hay_movimiento or yolo_ve or self.bloqueado or self.inicio is None:
            return False

        if (ahora - self.inicio) >= ESPERA_MOG2_S:
            ## Solo un evento MOG2 por episodio continuo de movimiento
            self.bloqueado = True
            return True

        return False

## PROCESAMIENTO YOLO
def letterbox_yolo(frame):
    ## Reescala a IMG_SIZE x IMG_SIZE sin deformar, rellenando con gris
    alto, ancho = frame.shape[:2]
    escala = min(IMG_SIZE / ancho, IMG_SIZE / alto)
    nuevo_ancho = int(round(ancho * escala))
    nuevo_alto = int(round(alto * escala))

    imagen = cv2.resize(frame, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LINEAR)

    pad_ancho = IMG_SIZE - nuevo_ancho
    pad_alto = IMG_SIZE - nuevo_alto
    izquierda = pad_ancho // 2
    arriba = pad_alto // 2

    imagen = cv2.copyMakeBorder(imagen, arriba, pad_alto - arriba, izquierda, pad_ancho - izquierda, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return imagen, escala, izquierda, arriba

def preparar_yolo(frame):
    imagen, escala, pad_x, pad_y = letterbox_yolo(frame)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen = imagen.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0

    return np.ascontiguousarray(imagen), escala, pad_x, pad_y

def ajustar_caja_yolo(x1, y1, x2, y2, escala, pad_x, pad_y, ancho, alto):
    ## Regresa la caja del espacio del modelo al espacio de la imagen original
    x1, x2 = (x1 - pad_x) / escala, (x2 - pad_x) / escala
    y1, y2 = (y1 - pad_y) / escala, (y2 - pad_y) / escala

    return (int(max(0, min(ancho - 1, x1))), int(max(0, min(alto - 1, y1))),
            int(max(0, min(ancho - 1, x2))), int(max(0, min(alto - 1, y2))))

def dibujar_yolo(frame, detecciones):
    for x1, y1, x2, y2, nombre, conf, _ in detecciones:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{nombre} {conf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    return frame

def extraer_detecciones(resultado, escala, pad_x, pad_y, ancho, alto):
    ## Filtra por confianza y clase, cada deteccion: (x1, y1, x2, y2, nombre, confianza, clase)
    detecciones = []

    for x1, y1, x2, y2, conf, clase in resultado:
        clase = int(clase)

        if conf < CONF_YOLO or clase not in CLASES:
            continue

        caja = ajustar_caja_yolo(x1, y1, x2, y2, escala, pad_x, pad_y, ancho, alto)
        detecciones.append((*caja, CLASES[clase], float(conf), clase))

    return detecciones

## Datos compartidos (con candado) entre el ciclo de camara y el hilo YOLO
class EstadoYolo:
    def __init__(self):
        self.lock = threading.Lock()
        self.nuevo_frame = threading.Event()
        self.frame = None
        self.detecciones = []
        self.tiempo_ms = 0.0
        self.fps = 0.0
        self.evento = None    ## (frame_anotado, nombres, detalles) pendiente de atender

    ## Lado camara
    def entregar_frame(self, frame):
        with self.lock:
            self.frame = frame
        self.nuevo_frame.set()

    def leer(self):
        ## Devuelve detecciones, ms, fps y el evento pendiente (que se consume)
        with self.lock:
            evento, self.evento = self.evento, None
            return list(self.detecciones), self.tiempo_ms, self.fps, evento

    ## Lado YOLO
    def tomar_frame(self):
        with self.lock:
            return self.frame

    def publicar(self, detecciones, tiempo_s, evento=None):
        with self.lock:
            self.detecciones = detecciones
            self.tiempo_ms = tiempo_s * 1000.0
            self.fps = 1.0 / tiempo_s if tiempo_s > 0 else 0.0

            if evento is not None:
                self.evento = evento

## HILO YOLO
def ejecutar_yolo(sesion, nombre_entrada, nombre_salida, estado, detener_yolo):
    while not detener_yolo.is_set():
        if not estado.nuevo_frame.wait(timeout=0.2):
            continue

        estado.nuevo_frame.clear()

        ## No se copia el frame: el ciclo principal nunca lo modifica
        frame = estado.tomar_frame()
        if frame is None:
            continue

        alto, ancho = frame.shape[:2]
        entrada_modelo, escala, pad_x, pad_y = preparar_yolo(frame)

        ## INFERENCIA
        inicio = time.perf_counter()
        resultado = sesion.run([nombre_salida], {nombre_entrada: entrada_modelo})[0][0]
        tiempo = time.perf_counter() - inicio

        detecciones = extraer_detecciones(resultado, escala, pad_x, pad_y, ancho, alto)

        ## Evento solo si hay algun polinizador (la flor NO cuenta)
        polinizadores = [d for d in detecciones if d[6] in CLASES_EVENTO]
        evento = None

        if polinizadores:
            nombres = list(dict.fromkeys(d[4] for d in polinizadores))
            detalles = [(d[4], d[5]) for d in polinizadores]
            evento = (dibujar_yolo(frame.copy(), detecciones), nombres, detalles)

        estado.publicar(detecciones, tiempo, evento)

def iniciar_yolo(estado, detener_yolo):
    ## Carga el modelo, lo calienta y lanza el hilo. Devuelve el hilo o None si falla
    try:
        print("Cargando modelo YOLO ONNX...", flush=True)
        opciones = ort.SessionOptions()
        opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        sesion = ort.InferenceSession(RUTA_MODELO_YOLO, sess_options=opciones, providers=["CPUExecutionProvider"])
        nombre_entrada = sesion.get_inputs()[0].name
        nombre_salida = sesion.get_outputs()[0].name

        ## Calentamiento del modelo
        print("Calentando modelo YOLO...", flush=True)
        dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)

        for _ in range(3):
            sesion.run([nombre_salida], {nombre_entrada: dummy})

        print("YOLO listo", flush=True)

        hilo = threading.Thread(target=ejecutar_yolo, args=(sesion, nombre_entrada, nombre_salida, estado, detener_yolo), daemon=True)
        hilo.start()
        return hilo

    except Exception as e:
        print("ERROR cargando YOLO:", e, flush=True)
        print("El sistema continuara solo con MOG2", flush=True)
        return None

## GRABACION DE VIDEO Y GUARDADO DE EVENTOS
class GrabadorVideo:
    def __init__(self):
        self.fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.vw = None
        self.ruta = ""
        self.origen = ""
        self.fin = 0.0

    @property
    def activo(self):
        return self.vw is not None

    def iniciar(self, origen, frame):
        ruta = os.path.join(RUTA_EVENTOS, f"evento_vid_{timestamp_nombre()}_{origen.lower()}.avi")
        alto, ancho = frame.shape[:2]
        vw = cv2.VideoWriter(ruta, self.fourcc, FPS_VIDEO, (ancho, alto))

        if not vw.isOpened():
            print("No se pudo iniciar VideoWriter", flush=True)
            return

        self.vw, self.ruta, self.origen = vw, ruta, origen
        self.fin = time.time() + DURACION_VIDEO
        print(f"Grabando video de evento {origen}: {ruta}", flush=True)

    def escribir(self, frame):
        if self.activo:
            self.vw.write(frame)

            if time.time() >= self.fin:
                self.cerrar()

    def cerrar(self):
        if self.activo:
            self.vw.release()
            print(f"Video de evento {self.origen} terminado", flush=True)

        self.vw, self.ruta, self.origen = None, "", ""

def guardar_imagen_evento(origen, frame_evento, nombres, detalles):
    ## Guarda la imagen del evento, devuelve los datos para el CSV (o None si fallo)
    fecha = datetime.now()
    ruta = os.path.join(RUTA_EVENTOS, f"evento_img_{fecha.strftime('%d-%m-%Y_%H%M%S')}_{origen.lower()}.jpg")

    if not cv2.imwrite(ruta, frame_evento):
        return None

    print("\nEVENTO DETECTADO", flush=True)
    print(f"Origen: {origen}", flush=True)

    if origen == "YOLO" and nombres:
        print("Clases:", ", ".join(nombres), flush=True)
    print(f"Imagen: {ruta}", flush=True)

    return fecha, origen, (detalles if origen == "YOLO" else []), ruta

def dibujar_preview(frame, yolo_fps, yolo_ms, estado, color):
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"YOLO: {yolo_fps:.2f} FPS", (10, 25), fuente, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"Inferencia: {yolo_ms:.0f} ms", (10, 50), fuente, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, "YOLO principal | MOG2 respaldo", (10, 75), fuente, 0.50, (255, 255, 255), 2)
    cv2.putText(frame, f"Estado: {estado}", (10, 100), fuente, 0.50, color, 2)
    cv2.imshow(NOMBRE_VENTANA, frame)

## MODO 2: YOLO + MOG2 (RESPALDO)
def modo_deteccion():
    print("Modo deteccion iniciado", flush=True)

    ## Abrir camara
    cap = abrir_cam(CAM_DETECCION, FPS_CAM)
    if cap is None:
        print("No se pudo abrir la camara de deteccion", flush=True)
        return

    ## Inicializar YOLO
    estado = EstadoYolo()
    detener_yolo = threading.Event()
    hilo_yolo = iniciar_yolo(estado, detener_yolo) if RED_ACTIVAR else None
    yolo_activo = hilo_yolo is not None

    ## Inicializar MOG2: se activa por configuracion o si YOLO no esta disponible
    mog2 = None
    if USAR_MOG2_RESPALDO or not yolo_activo:
        mog2 = RespaldoMog2()
        print("Calibrando fondo MOG2", flush=True)
        mog2.calibrar(cap, FPS_CAM)
        print("Fondo MOG2 calibrado", flush=True)

    ## Control de eventos y video
    video = GrabadorVideo()
    ultimo_guardado = 0.0

    ## Ciclo principal
    try:
        while not detener.is_set():
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el frame de deteccion", flush=True)
                time.sleep(0.2)
                continue

            ## Entregar frame reciente a YOLO
            if yolo_activo:
                estado.entregar_frame(frame)

            ## Procesar MOG2
            cajas = mog2.procesar(frame) if mog2 else []
            hay_movimiento = len(cajas) > 0

            ## Leer resultado mas reciente de YOLO
            if yolo_activo:
                detecciones, yolo_ms, yolo_fps, evento_yolo = estado.leer()
            else:
                detecciones, yolo_ms, yolo_fps, evento_yolo = [], 0.0, 0.0, None

            ## Saber si YOLO sigue viendo un polinizador
            yolo_ve = any(d[6] in CLASES_EVENTO for d in detecciones)

            ## Frame visual (el que se muestra y se graba en video)
            frame_visual = dibujar_yolo(frame.copy(), detecciones)
            ahora = time.time()

            ## Control de respaldo de MOG2
            if mog2:
                mog2.actualizar(hay_movimiento, ahora)

                ## Mostrar cajas solo cuando MOG2 actua como respaldo
                if hay_movimiento and not yolo_ve and evento_yolo is None:
                    dibujar_mog2(frame_visual, cajas)

            ## Decision del evento
            evento = False
            origen = ""
            frame_evento = None
            nombres, detalles = [], []

            ## 1. Deteccion de evento con YOLO
            if evento_yolo is not None:
                frame_evento, nombres, detalles = evento_yolo
                evento = True
                origen = "YOLO"

                ## Este episodio de movimiento ya fue atendido por YOLO
                if mog2:
                    mog2.bloqueado = True

            ## 2. MOG2 actua como respaldo
            elif mog2 and mog2.debe_disparar(hay_movimiento, yolo_ve, ahora):
                evento = True
                origen = "MOG2"
                frame_evento = dibujar_mog2(frame.copy(), cajas)

            ## Guardar imagen
            registro = None
            if evento and (ahora - ultimo_guardado) >= COOLDOWN:
                registro = guardar_imagen_evento(origen, frame_evento, nombres, detalles)
                ultimo_guardado = ahora

            ## Iniciar video
            if evento and not video.activo:
                video.iniciar(origen, frame_visual)

            ## Registrar en el CSV (con el video asociado si se esta grabando)
            if registro is not None:
                registrar_evento(*registro, video.ruta)

            ## Grabar video
            video.escribir(frame_visual)

            ## Preview
            if MOSTRAR_PREVIEW:
                if yolo_ve:
                    texto, color = "YOLO detectando", (0, 255, 0)
                elif mog2 and mog2.bloqueado:
                    texto, color = "MOG2 bloqueado", (0, 255, 255)
                elif hay_movimiento:
                    texto, color = "MOG2 esperando YOLO", (0, 255, 255)
                else:
                    texto, color = "monitoreando", (255, 255, 255)

                dibujar_preview(frame_visual, yolo_fps, yolo_ms, texto, color)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Preview cerrado por usuario", flush=True)
                    break

            time.sleep(0.001)

    finally:
        ## Detener hilo YOLO
        detener_yolo.set()
        estado.nuevo_frame.set()

        if hilo_yolo is not None:
            hilo_yolo.join(timeout=3)

        video.cerrar()

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
    print(f"Modelo YOLO: {RUTA_MODELO_YOLO}", flush=True)
    print(f"Resolucion YOLO: {IMG_SIZE}x{IMG_SIZE}", flush=True)
    print(f"Confianza YOLO: {CONF_YOLO}", flush=True)
    print(f"YOLO activado: {RED_ACTIVAR}", flush=True)
    print(f"MOG2 respaldo: {USAR_MOG2_RESPALDO}", flush=True)
    print(f"Espera MOG2: {ESPERA_MOG2_S} s", flush=True)
    print(f"Rearme MOG2: {REARME_MOG2_S} s", flush=True)

    crear_carpetas()

    ## HILO CON LOS 2 SUBSISTEMAS PRINCIPALES
    hilos = [threading.Thread(target=modo_foto), threading.Thread(target=modo_deteccion)]

    for h in hilos:
        h.start()
    try:
        for h in hilos:
            h.join()
    except KeyboardInterrupt:
        print("\nDeteniendo el sistema...", flush=True)
        detener.set()
        for h in hilos:
            h.join()

if __name__ == "__main__":
    main()
