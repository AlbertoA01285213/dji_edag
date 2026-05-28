#!/usr/bin/env python3
import os
import cv2
import time
import sqlite3
import threading
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from ultralytics import YOLO
import zxingcpp

app = FastAPI(title="YOLO Video Analyzer Server")

# Variables globales para almacenar el estado en tiempo real
ESTADO_PROCESO = {
    "corriendo": False,

    "data_barcode": "—",
    "data_qr": "—",
    "data_vin": "—",
    "data_datamax": "—",

    "data_tiempo_procesamiento": "—",
    "data_num_frame": "—",
    "data_num_frame_max": "—",

    "formato": "—",
    "texto": "Esperando scanner...",

    "frame": None,
    "recorte_etiqueta": None,
    "recorte_barcode": None,
    "recorte_qr": None,
    "recorte_vin": None,
    "recorte_datamax": None
}

DEBE_PARAR = False
CONFIDENCE_THRESHOLD = 0.8

# Configuraciones de rutas fijas de tu proyecto
PATH_BASE = os.path.join(os.path.expanduser('~'), 'Documents', 'dji_edag', 'reconocimiento')
PATH_MODELO_ETIQUETA = os.path.join(PATH_BASE, 'modelos', 'modelo_etiqueta.pt')
PATH_MODELO_CODIGOS = os.path.join(PATH_BASE, 'modelos', 'modelo_codigos.pt')
PATH_OUTPUT = os.path.join(PATH_BASE, 'output')

# Cargar modelos globalmente al encender el servidor
print("Cargando modelos YOLO en el Servidor...")
modelo_etiqueta = YOLO(PATH_MODELO_ETIQUETA)
modelo_codigos = YOLO(PATH_MODELO_CODIGOS)


class VideoPayload(BaseModel):
    ruta_video: str


def optimizar_y_convertir_base64(img, max_width=640):
    """Convierte un frame de OpenCV a string Base64 optimizado para transmitir por HTTP rápidamente"""
    if img is None or img.size == 0:
        return None
    # Resizear para no saturar la red local con imágenes 4K o Full HD
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)))
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    import base64
    return base64.b64encode(buffer).decode('utf-8')

def analizar_imagen(img, modelo, usar_tracking=False, frame_num=0):
    """Ejecuta YOLO (con soporte opcional para ByteTrack) y devuelve un diccionario limpio"""
    if usar_tracking:
        persistir = True if frame_num % 3 == 0 else False
        results = modelo.track(source=img, persist=persistir, tracker="bytetrack.yaml", verbose=False)[0]
    else:
        results = modelo(img, verbose=False)[0]

    detections = {}
    masks = results.masks

    if results.boxes is None:
        return detections

    for i, box in enumerate(results.boxes):
        conf = float(box.conf[0])
        if conf < CONFIDENCE_THRESHOLD:
            continue

        label = modelo.names[int(box.cls[0])]
        
        # Recuperar el track_id si el tracker está activo
        track_id = int(results.boxes.id[i].item()) if (results.boxes.id is not None) else None

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        w, h = x2 - x1, y2 - y1
        cx = int(x1 + w / 2)
        cy = int(y1 + h / 2)
        
        mask = None
        if masks is not None:
            mask = masks.data[i].cpu().numpy()

        det_data = {
            'bbox': [int(x1), int(y1), int(w), int(h)],
            'center': (cx, cy),
            'conf': conf,
            'mask': mask,
            'track_id': track_id
        }

        if label not in detections:
            detections[label] = []
        detections[label].append(det_data)

    return detections

def dibujar_boxes(img, resultados):
    debug_img = img.copy()
    
    # CONTROL DE DAÑOS: Si llega una detección única por error, la envolvemos en el formato correcto
    if isinstance(resultados, dict) and 'bbox' in resultados:
        resultados = {"etiqueta_detectada": [resultados]}

    for label, objs in resultados.items():
        # Nos aseguramos de que objs sea una lista iterable
        if not isinstance(objs, list):
            continue
            
        for det in objs:
            # Nos aseguramos de que det sea el diccionario con los datos reales
            if not isinstance(det, dict) or 'bbox' not in det:
                continue

            color = (0, 255, 0)
            x, y, w, h = det['bbox']
            cx, cy = det['center']
            conf = det['conf']
            mask = det['mask']

            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
            texto = f"{label}: {conf:.2f}"
            cv2.putText(debug_img, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if mask is not None:
                mask_uint8 = (mask * 255).astype("uint8")
                mask_resized = cv2.resize(mask_uint8, (debug_img.shape[1], debug_img.shape[0]))
                colored_mask = np.zeros_like(debug_img)
                colored_mask[mask_resized > 0] = color
                
                debug_img = cv2.addWeighted(debug_img, 1.0, colored_mask, 0.4, 0)

    return debug_img

def extraer_todas_las_detecciones(detecciones_dict):
    """Reúne todos los diccionarios de detección en una lista plana"""
    lista_plana = []
    for lista_objetos in detecciones_dict.values():
        lista_plana.extend(lista_objetos)
    return lista_plana

def obtener_mejor_deteccion(detecciones_dict, label):
    """Busca de forma segura el objeto de mayor confianza asociado a una etiqueta específica"""
    lista_objetos = detecciones_dict.get(label, [])
    if not lista_objetos:
        return None
    return max(lista_objetos, key=lambda x: x['conf'])

def recortar_frame(img, bbox):
    """Recorta de forma segura una región pasándole directamente el bounding box [x, y, w, h]"""
    x, y, w, h = bbox
    # Límites mapeados matemáticamente para no desbordar las dimensiones de la matriz
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    return img[y1:y2, x1:x2]
                

def bucle_vision_artificial(video_path):
    """Tu script original de procesamiento de video adaptado a la API"""
    global ESTADO_PROCESO, DEBE_PARAR
    ESTADO_PROCESO["corriendo"] = True
    
    etiquetas_procesadas = set()
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened() and not DEBE_PARAR:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        t_init = time.time()

        data_img_analizada = analizar_imagen(frame, modelo_etiqueta, usar_tracking=True, frame_num=frame_num)

        lista_etiquetas = extraer_todas_las_detecciones(data_img_analizada)
        if not lista_etiquetas:
            continue

        etiqueta_encontrada = max(lista_etiquetas, key=lambda x: x['conf'])

        frame_etiqueta = dibujar_boxes(frame, data_img_analizada)                     # Se dibuja la caja de la etiqueta en la imagen completa
        img_etiqueta = recortar_frame(frame, etiqueta_encontrada['bbox']) 
        
        if img_etiqueta.size == 0:
            continue

        data_codigos_analizados = analizar_imagen(img_etiqueta, modelo_codigos)

        barcode_det = obtener_mejor_deteccion(data_codigos_analizados, 'VIN')
        qr_det = obtener_mejor_deteccion(data_codigos_analizados, 'PKN_Largo')
        vin_det = obtener_mejor_deteccion(data_codigos_analizados, 'CVE_COM')
        datamax_det = obtener_mejor_deteccion(data_codigos_analizados, 'DATAMATRIX_NUM')

        data_barcode, data_qr, data_vin, data_datamax = "—", "—", "—", "—"
        img_barcode_b64, img_qr_b64, img_vin_b64, img_datamax_b64 = None, None, None, None

        if barcode_det:
            crop = recortar_frame(img_etiqueta, barcode_det['bbox'])
            img_barcode_b64 = optimizar_y_convertir_base64(crop, 280)
            scan = zxingcpp.read_barcode(crop)
            if scan and scan.text: data_barcode = scan.text

        if qr_det:
            crop = recortar_frame(img_etiqueta, qr_det['bbox'])
            img_qr_b64 = optimizar_y_convertir_base64(crop, 280)
            scan = zxingcpp.read_barcode(crop)
            if scan and scan.text: data_qr = scan.text

        if vin_det:
            crop = recortar_frame(img_etiqueta, vin_det['bbox'])
            img_vin_b64 = optimizar_y_convertir_base64(crop, 280)
            scan = zxingcpp.read_barcode(crop)
            if scan and scan.text: data_vin = scan.text

        if datamax_det:
            crop = recortar_frame(img_etiqueta, datamax_det['bbox'])
            img_datamax_b64 = optimizar_y_convertir_base64(crop, 280)
            scan = zxingcpp.read_barcode(crop)
            if scan and scan.text: data_datamax = scan.text


        t_fin = time.time() - t_init

        ESTADO_PROCESO["frame"] = optimizar_y_convertir_base64(frame, 640)
        ESTADO_PROCESO["recorte_etiqueta"] = optimizar_y_convertir_base64(frame_etiqueta, 280)
        ESTADO_PROCESO["recorte_barcode"] = img_barcode_b64
        ESTADO_PROCESO["recorte_qr"] = img_qr_b64
        ESTADO_PROCESO["recorte_vin"] = img_vin_b64
        ESTADO_PROCESO["recorte_datamax"] = img_datamax_b64

        ESTADO_PROCESO["data_barcode"] = data_barcode
        ESTADO_PROCESO["data_qr"] = data_qr
        ESTADO_PROCESO["data_vin"] = data_vin
        ESTADO_PROCESO["data_datamax"] = data_datamax

        ESTADO_PROCESO["data_tiempo_procesamiento"] = f"{round(t_fin * 1000, 1)} ms"
        ESTADO_PROCESO["data_num_frame"] = int(frame_num)
        ESTADO_PROCESO["data_num_frame_max"] = int(total_frames)
        

    cap.release()
    ESTADO_PROCESO["corriendo"] = False
    print("Análisis de video terminado.")

@app.post("/iniciar")
def iniciar_analisis(payload: VideoPayload, background_tasks: BackgroundTasks):
    """Endpoint al que el Dashboard llamará para arrancar el procesamiento"""
    if ESTADO_PROCESO["corriendo"]:
        return {"status": "error", "message": "Ya hay un análisis en ejecución."}
    
    # Ejecutar en un hilo de fondo (Background Task) para que la petición HTTP responda de inmediato
    background_tasks.add_task(bucle_vision_artificial, payload.ruta_video)
    return {"status": "success", "message": "Análisis iniciado correctamente."}

@app.post("/parar")
def parar_analisis():
    global DEBE_PARAR
    if not ESTADO_PROCESO["corriendo"]:
        return {"status": "error", "message": "No hay ningún análisis activo que detener."}
    
    DEBE_PARAR = True
    return {"status": "success", "message": "Se ha enviado la señal de parada al analizador."}

@app.get("/estado")
def obtener_estado():
    """Endpoint al que el Dashboard llamará repetidamente para refrescar su pantalla"""
    return ESTADO_PROCESO

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)