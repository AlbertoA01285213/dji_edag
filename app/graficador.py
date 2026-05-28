#!/usr/bin/env python3
import os
import re
import io
import cv2
import time
import logging
import sqlite3
import threading
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import base64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Graficador3D")

app = FastAPI(title="YOLO Graph Analyzer Server")

ESTADO_PROCESO = {
    "corriendo": False,
    "formato": "—",
    "texto": "Esperando scanner...",
    "grafica": None,
    "frame_actual_base64": None
}

DEBE_PARAR = False

class WaypointPayload(BaseModel):
    ruta_waypoint: str


class GeneradorGrafica3D:
    def __init__(self, trayectoria_dict):
        self.trayectoria = trayectoria_dict
        # Extraemos las coordenadas completas una sola vez para ahorrar procesamiento
        self.x = [pt[0] for pt in self.trayectoria.values()]
        self.y = [pt[1] for pt in self.trayectoria.values()]
        self.z = [pt[2] for pt in self.trayectoria.values()]
        logger.info(f"Generador Grafica 3D inicializado con {len(self.x)} puntos de trayectoria.")

    def generar_frame_b64(self, frame_num):
        """Dibuja la ruta, remarca el frame actual y exporta a Base64 sin abrir ventanas"""
        # Crear figura con el mismo color de tu interfaz (#2c3e50)
        fig = plt.figure(figsize=(5, 4), dpi=100)
        fig.patch.set_facecolor('#2c3e50')
        
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#2c3e50')
        
        # 1. Dibujar la línea de trayectoria completa (Turquesa)
        ax.plot(self.x, self.y, self.z, color='#1abc9c', linewidth=2)
        
        # 2. BONUS: Remarcar el punto actual en Rojo Neón si existe en el SRT
        if frame_num in self.trayectoria:
            cx, cy, cz = self.trayectoria[frame_num]
            ax.scatter([cx], [cy], [cz], color='#e74c3c', s=100, edgecolors='white', zorder=5)
        
        # Estilizar el mapa oscuro
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        ax.set_zlabel('Altitud (m)')
        ax.grid(True, color='#7f8c8d')
        
        # 3. Guardar la gráfica directamente en la memoria RAM (BytesIO)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        buf.seek(0)
        
        # Convertir a Base64 string
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        
        # IMPORTANTE: Cerrar la figura para liberar memoria RAM del servidor
        plt.close(fig)
        
        return img_b64
    

def parsear_srt_dji(ruta_srt):
    """Lee el archivo SRT del dron y devuelve un mapa de {frame_num: (lon, lat, alt)}"""
    trayectoria = {}
    if not os.path.exists(ruta_srt):
        logger.warning(f"No se encontró el archivo '{ruta_srt}'. Generando ruta simulada de respaldo...")
        print(f"Aviso: No se encontró el archivo '{ruta_srt}'. Se generará una ruta simulada.")
        # Generamos una ruta simulada en espiral si no existe el archivo físico
        for f in range(1, 500):
            trayectoria[f] = (-100.266 + np.sin(f/20)*0.001, 25.584 + np.cos(f/20)*0.001, 2.4 + f*0.05)
        return trayectoria

    try:
        with open(ruta_srt, 'r', encoding='utf-8') as f:
            contenido = f.read()
        
        bloques = contenido.split('\n\n')
        for bloque in bloques:
            frame_match = re.search(r"FrameCnt:\s*(\d+)", bloque)
            lat_match = re.search(r"latitude:\s*([\d.-]+)", bloque)
            lon_match = re.search(r"longitude:\s*([\d.-]+)", bloque)
            alt_match = re.search(r"rel_alt:\s*([\d.-]+)", bloque)
            
            if frame_match and lat_match and lon_match and alt_match:
                f_num = int(frame_match.group(1))
                trayectoria[f_num] = (float(lon_match.group(1)), float(lat_match.group(1)), float(alt_match.group(1)))

        logger.info(f"Parseo exitoso. Se detectaron {len(trayectoria)} frames con telemetría válida.")
    except Exception as e:
        logger.error(f"Error crítico al leer el archivo SRT: {e}", exc_info=True)
        print(f"Error al leer el archivo SRT: {e}")
        
    return trayectoria


def bucle_waypoint(waypoint_path):
    global ESTADO_PROCESO, DEBE_PARAR
    logger.info("Iniciando bucle de procesamiento en segundo plano...")
    ESTADO_PROCESO["corriendo"] = True
    ESTADO_PROCESO["frame_actual_base64"] = "procesando"
    
    datos_vuelo = parsear_srt_dji(waypoint_path)
    if not datos_vuelo:
        logger.error("No se pudieron cargar coordenadas del SRT. Cancelando hilo.")
        ESTADO_PROCESO["corriendo"] = False
        return

    generador = GeneradorGrafica3D(datos_vuelo)
    frames_ordenados = sorted(datos_vuelo.keys())

    for i,frame_num in enumerate(frames_ordenados):
        if DEBE_PARAR:
            logger.warning("Señal de parada interceptada. Saliendo del bucle prematuramente.")
            print("Señal de parada recibida en el graficador.")
            break
            
        t_inicio = time.time()

        try:
            base64_grafica = generador.generar_frame_b64(frame_num)
            ESTADO_PROCESO["grafica"] = base64_grafica

            if i % 10 == 0:
                logger.info(f"Procesando gráfico de telemetría para Frame ID: {frame_num} ({i+1}/{len(frames_ordenados)})")
        except Exception as e:
            logger.error(f"Error procesando el Frame {frame_num}: {e}")
        
        
        tiempo_render = time.time() - t_inicio
        delay_tolerancia = max(0.001, 0.033 - tiempo_render)
        time.sleep(delay_tolerancia)

    # Al finalizar restablecemos los controles de estado
    ESTADO_PROCESO["corriendo"] = False
    # Dejamos un valor no nulo al terminar para que se active tu QMessageBox en el frontend
    ESTADO_PROCESO["frame_actual_base64"] = "finalizado"


# from fastapi import Response

# @app.get("/grafica_en_vivo")
# def ver_grafica_en_navegador():
#     """Endpoint de diagnóstico para ver la imagen real desde cualquier navegador"""
#     # Si no hay gráfica aún o el proceso no ha iniciado
#     if not ESTADO_PROCESO["grafica"]:
#         return {"status": "esperando", "message": "Inicia el análisis en el dashboard primero para generar la imagen."}
    
#     try:
#         # Decodificamos el string Base64 actual a bytes puros de una imagen PNG
#         imagen_bytes = base64.b64decode(ESTADO_PROCESO["grafica"])
        
#         # Le respondemos al navegador con los bytes crudos de la imagen y el formato correcto
#         return Response(content=imagen_bytes, media_type="image/png")
#     except Exception as e:
#         return {"status": "error", "message": f"No se pudo decodificar la imagen: {e}"}
    

@app.post("/iniciar")
def iniciar_grafica(payload: WaypointPayload, background_tasks: BackgroundTasks):
    logger.info(f"Petición POST /iniciar recibida. Ruta proporcionada: {payload.ruta_waypoint}")
    if ESTADO_PROCESO["corriendo"]:
        logger.warning("Intento de inicio denegado: Ya hay un análisis en curso.")
        return {"status": "error", "message": "Ya hay un analisis en ejecucion"}
    
    DEBE_PARAR = False

    background_tasks.add_task(bucle_waypoint, payload.ruta_waypoint)
    logger.info("Tarea en segundo plano (Background Task) registrada correctamente.")
    return {"status": "success", "message": "Analisis iniciado correctamente"}

@app.post("/parar")
def parar_analisis():
    global DEBE_PARAR
    if not ESTADO_PROCESO["corriendo"]:
        return {"status": "error", "message": "No hay ningún análisis activo que detener."}
    
    DEBE_PARAR = True
    return {"status": "success", "message": "Se ha enviado la señal de parada al analizador."}


@app.get("/estado")
def obtener_estado():
    return ESTADO_PROCESO
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.2", port=8001)