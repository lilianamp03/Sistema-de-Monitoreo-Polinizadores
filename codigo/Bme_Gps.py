## Librerias para BME688
import time, bme680

## Librerias para GPS
import serial, pynmea2

## Librerias para CSV / JSON
import csv, os, json
from datetime import datetime

ruta = "/home/pi/Desktop/Monitoreo/Datos"
ruta_panel = "/home/pi/Desktop/Monitoreo/Panel/estado.json"
ruta_panel_ap ="/home/pi/Desktop/Archivos_public/estado.json"
encabezados = ["time", "temp", "hume", "pres", "gas", "lat", "lon"]

def leer_bme(address = 0x77):
    try:
        ## Inicializa el sensor 
        bme = bme680.BME680(i2c_addr = address)
        ## Activar la medicion del sensor de gas
        bme.set_gas_status(bme680.ENABLE_GAS_MEAS)
        ## Configura la temperatura del calentador interno
        bme.set_gas_heater_temperature(320)
        ## Definir cuanto tiempo estara activo el calentador antes de medir
        bme.set_gas_heater_duration(150)
        ## Pefil de calentamiento clasico
        bme.select_gas_heater_profile(0)
        
        ## Asegurar que el sensor de gas este a su temperatura interna correcta 
        for i in range(10):
            bme.get_sensor_data()
            
            if bme.data.heat_stable:
                break
            time.sleep(0.5)
            
        return {
            "temp": round(bme.data.temperature, 2),
            "hume": round(bme.data.humidity,2),
            "pres": round(bme.data.pressure, 2),
            "gas": round(bme.data.gas_resistance,2),
            ## "heat_stable": bme.data.heat_stable
        }
    
    except Exception as e:
        print("Error al leer BME688: ", e, flush=True)
        
        return {
            "temp": None,
            "hume": None,
            "pres": None,
            "gas": None,
            ## "heat_stable": False
        }

def imprimir_bme(datos):
    print("Temperatura:\t", datos["temp"], "C", flush=True)
    print("Humedad:\t", datos["hume"], "%", flush=True)
    print("Presion:\t", datos["pres"], "hPa", flush=True)
    print("Gas:\t\t", datos["gas"], "Ohm", flush=True)
    ## print("Gas estable:\t", datos["heat_stable"], flush=True)
    
def leer_gps(tiempo_max=20):
    inicio = time.monotonic()  ## Guarda el tiempo inicial 
    buffer = "" ## Buffer donde se iran acumulando los datos que lleguen al GPS

    try:
        ## Abrir el puerto serial
        with serial.Serial("/dev/serial0", 9600, timeout=0.2) as gps:
            gps.reset_input_buffer() ## Limpia el buffer interno del puerto serial, elimina datos viejos

            while time.monotonic() - inicio < tiempo_max:
                datos = gps.read(128).decode("ascii", "ignore") ## Lee hasta 128 bytes del puerto serial

                buffer += datos ## Agrega los datos recibidos al buffer

                while "\n" in buffer:	## Verifica si hay salto de linea en buffer, si hay, significa que hay al menos una linea completa de GPS
                    linea, buffer = buffer.split("\n", 1) ## Separa buffer en 2 partes, linea= primera linea completa encontrada
                    linea = linea.strip()

                    if not linea:
                        continue

                    if linea.startswith(("$GPGGA", "$GNGGA", "$GPRMC", "$GNRMC")):
                        try:
                            m = pynmea2.parse(linea)

                            if m.sentence_type == "GGA":
                                if m.latitude!=0 and m.longitude!=0:
                                    return {
                                        "lat": m.latitude,
                                        "lon": m.longitude
                                    }

                        except Exception as e:
                            print("Error GPS:", e, flush=True)

    except Exception as e:
        print("Error al leer GPS:", e, flush=True)

    print("GPS sin coordenadas validas", flush=True)

    return {
        "lat": None,
        "lon": None
    }
                        
def imprimir_gps(datos):
    print("Latitud:\t", datos["lat"], flush=True)
    print("Longitud:\t", datos["lon"], flush=True)
    
def guardar_csv(fecha_hora, datos_bme, datos_gps):
    os.makedirs(ruta, exist_ok=True)
    
    nombre_archivo = "mediciones_" + datetime.now().strftime("%d-%m-%Y") + ".csv"
    archivo = os.path.join(ruta, nombre_archivo)
    
    existe = os.path.exists(archivo)
    
    
    with open(archivo, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        if not existe:
            writer.writerow(encabezados)
            
        writer.writerow([
            fecha_hora,
            datos_bme["temp"],
            datos_bme["hume"],
            datos_bme["pres"],
            datos_bme["gas"],
            datos_gps["lat"],
            datos_gps["lon"]
        ])
        
def guardar_json(fecha_hora, datos_bme, datos_gps):
    os.makedirs(ruta, exist_ok=True)
    
    nombre = "mediciones_" + datetime.now().strftime("%d-%m-%Y") + ".jsonl"
    archivo = os.path.join(ruta, nombre)
    
    registro = {
        "time": fecha_hora,
        "temp": datos_bme["temp"],
        "hume": datos_bme["hume"],
        "pres": datos_bme["pres"],
        "gas": datos_bme["gas"],
        "lat": datos_gps["lat"],
        "lon": datos_gps["lon"]
    }
    
    with open(archivo, "a") as f:
        f.write(json.dumps(registro) + "\n")
 
def guardar_estado_panel(fecha_hora, datos_bme, datos_gps):
    carpeta_panel = os.path.dirname(ruta_panel)
    os.makedirs(carpeta_panel, exist_ok=True)

    estado = {
        "fecha": fecha_hora,
        "temperatura": datos_bme["temp"],
        "humedad": datos_bme["hume"],
        "presion": datos_bme["pres"],
        "gas": datos_bme["gas"],
        "latitud": datos_gps["lat"],
        "longitud": datos_gps["lon"],
        "gps_valido": datos_gps["lat"] is not None and datos_gps["lon"] is not None
    }

    with open(ruta_panel, "w", encoding="utf-8") as f:
        json.dump(estado, f, indent=4)

    print("Estado del panel actualizado")
    
def guardar_estado_panel_ap(fecha_hora, datos_bme, datos_gps):
    carpeta_panel_ap = os.path.dirname(ruta_panel_ap)
    os.makedirs(carpeta_panel_ap, exist_ok=True)

    estado = {
        "fecha": fecha_hora,
        "temperatura": datos_bme["temp"],
        "humedad": datos_bme["hume"],
        "presion": datos_bme["pres"],
        "gas": datos_bme["gas"],
        "latitud": datos_gps["lat"],
        "longitud": datos_gps["lon"],
        "gps_valido": datos_gps["lat"] is not None and datos_gps["lon"] is not None
    }

    with open(ruta_panel_ap, "w", encoding="utf-8") as f:
        json.dump(estado, f, indent=4)

    print("Estado del panel actualizado")
    
## MAIN
if __name__ == "__main__":
    print("Iniciando script")
    fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    datos_bme = leer_bme(address = 0x77)
    datos_gps = leer_gps(tiempo_max=20)

    imprimir_bme(datos_bme)
    imprimir_gps(datos_gps)
    
    guardar_csv(fecha_hora, datos_bme, datos_gps)
    guardar_json(fecha_hora, datos_bme, datos_gps)
    guardar_estado_panel(fecha_hora, datos_bme, datos_gps)
    guardar_estado_panel_ap(fecha_hora, datos_bme, datos_gps)
    
    print("Datos guardados correctamente")
