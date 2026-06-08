#!/usr/bin/env python3
import os
import cv2
import time
import sqlite3
import logging
import threading
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from ultralytics import YOLO
import zxingcpp
from pylibdmtx.pylibdmtx import decode
from paddleocr import TextRecognition

app = FastAPI(title="YOLO Video Analyzer Server")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("Analizador")

# Variables globales para almacenar el estado en tiempo real
ESTADO_PROCESO = {
    "corriendo": False,

    "frame": None,

    "recorte_etiqueta": None,

    "recorte_vin_total": None,
    "recorte_vin_ult": None,
    "recorte_datamatrix_link": None,
    "recorte_datamatrix_num": None,
    "recorte_pkn_largo": None,
    "recorte_cve_com": None,
    "recorte_vin_barra": None,

    "vin_ult_data": "—",
    "datamatrix_link_data": "—",
    "datamatrix_num_data": "—",
    "pkn_largo_data": "—",
    "cve_com_data": "—",
    "vin_barra_data": "—",

    "data_tiempo_procesamiento": "—",
    "data_num_frame": "—",
    "data_num_frame_max": "—",

    "formato": "—",
    "texto": "Esperando scanner..."
}

MODEL = TextRecognition(model_name="PP-OCRv5_server_rec")

conn = sqlite3.connect("datos.db")
cursor = conn.cursor()
cursor.execute("""
               CREATE TABLE IF NOT EXISTS etiquetas (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               vin_barra TEXT,
               vin_ult TEXT,
               datamatrix_link TEXT,
               datamatrix_num TEXT,
               pkn_largo TEXT,
               cve_com TEXT,
               latitud TEXT,
               longitud TEXT,
               fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
               )
               """)
conn.commit()
conn.close()

DEBE_PARAR = False
ESTA_PAUSADO = False
CONFIDENCE_THRESHOLD = 0.8

# Configuraciones de rutas fijas de tu proyecto
PATH_BASE = os.path.join(os.path.expanduser('~'), 'Documents', 'dji_edag', 'app')
PATH_MODELO_ETIQUETA = os.path.join(PATH_BASE, 'modelos', 'modelo_etiqueta.pt')
PATH_MODELO_CODIGOS = os.path.join(PATH_BASE, 'modelos', 'modelo_codigos.pt')
PATH_OUTPUT = os.path.join(PATH_BASE, 'output')

# Cargar modelos globalmente al encender el servidor
print("Cargando modelos YOLO en el Servidor...")
modelo_etiqueta = YOLO(PATH_MODELO_ETIQUETA)
modelo_codigos = YOLO(PATH_MODELO_CODIGOS)


class VideoPayload(BaseModel):
    ruta_video: str

class VideoPause(BaseModel):
    pausa_video: int


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
    # if usar_tracking:
    #     persistir = True if frame_num % 3 == 0 else False
    #     results = modelo.track(source=img, persist=persistir, tracker="bytetrack.yaml", verbose=False)[0]
    if usar_tracking:
        # CORRECCIÓN: 'persist' DEBE ser siempre True en un flujo de video continuo
        results = modelo.track(
            source=img, 
            persist=True, 
            tracker="bytetrack.yaml", 
            verbose=False
        )[0]
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
        # track_id = int(results.boxes.id[i].item()) if (results.boxes.id is not None) else None
        track_id = None
        if usar_tracking and results.boxes.id is not None:
            try:
                track_id = int(results.boxes.id[i].item())
            except Exception:
                track_id = None
                
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

            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 4)
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

def guardar_etiqueta_en_db(datos):
    """Inserta de forma segura un registro consolidado en la BD aislando la conexión por hilo"""
    # Evitamos guardar registros completamente vacíos que no aporten información limpia
    if all(v == "—" for v in datos.values()):
        return

    try:
        conn = sqlite3.connect("datos.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO etiquetas (vin_barra, vin_ult, datamatrix_link, datamatrix_num, pkn_largo, cve_com)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datos["vin_barra_data"] if datos["vin_barra_data"] != "—" else None,
            datos["vin_ult_data"] if datos["vin_ult_data"] != "—" else None,
            datos["datamatrix_link_data"] if datos["datamatrix_link_data"] != "—" else None,
            datos["datamatrix_num_data"] if datos["datamatrix_num_data"] != "—" else None,
            datos["pkn_largo_data"] if datos["pkn_largo_data"] != "—" else None,
            datos["cve_com_data"] if datos["cve_com_data"] != "—" else None
        ))
        conn.commit()
        conn.close()
        logger.info("--> [DB SUCCESS] Se ha guardado una nueva etiqueta física en la Base de Datos.")
    except Exception as e:
        logger.error(f"Error crítico al intentar escribir en SQLite: {e}")

def ordenar_puntos(pts):
    """Ordena 4 puntos en orden estricto: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]"""
    pts = pts.reshape(4, 2)
    nuevo_orden = np.zeros((4, 2), dtype=np.float32)

    suma = pts.sum(axis=1)
    nuevo_orden[0] = pts[np.argmin(suma)]
    nuevo_orden[2] = pts[np.argmax(suma)]
    
    dif = np.diff(pts, axis=1).flatten()
    nuevo_orden[1] = pts[np.argmin(dif)]
    nuevo_orden[3] = pts[np.argmax(dif)]

    return nuevo_orden

def rectificar(img):
    """
    Detecta la inclinación de un código y lo rota para enderezarlo 
    SIN recortarlo, expandiendo el lienzo y manteniendo un fondo blanco.
    """
    if img is None or img.size == 0:
        return img, False, None

    h, w = img.shape[:2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    edges = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, False, None
    
    area_min = h * w * 0.10
    candidatos = [c for c in contours if cv2.contourArea(c) > area_min]
    if not candidatos:
        return img, False, None

    mayor = max(candidatos, key=cv2.contourArea)
    peri = cv2.arcLength(mayor, True)
    approx = cv2.approxPolyDP(mayor, 0.02 * peri, True)

    if len(approx) == 4:
        pts = ordenar_puntos(approx)
    else:
        rect = cv2.minAreaRect(mayor)
        box = cv2.boxPoints(rect)
        pts = ordenar_puntos(box)

    tl, tr, br, bl = pts

    angle = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))

    if abs(angle) < 0.5:
        return img, True, pts.astype(int)

    cX, cY = w // 2, h // 2

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    
    warped = cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return warped, True, pts.astype(int)

        
def bucle_vision_artificial(video_path):
    """Tu script original de procesamiento de video adaptado a la API"""
    global ESTADO_PROCESO, DEBE_PARAR
    ESTADO_PROCESO["corriendo"] = True
    DEBE_PARAR = False
    
    etiquetas_procesadas = set()
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    id_etiqueta_actual = None

    ultimos_datos = {
        "vin_ult_data": "—",
        "datamatrix_link_data": "—",
        "datamatrix_num_data": "—",
        "pkn_largo_data": "—",
        "cve_com_data": "—",
        "vin_barra_data": "—"
    }

    frame_null = np.zeros((280, 280, 3), dtype=np.uint8)
    frame_null = cv2.putText(frame_null, "Etiqueta no encontrada", (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    frame_null = optimizar_y_convertir_base64(frame_null, 280)


    while cap.isOpened() and not DEBE_PARAR:
        if ESTA_PAUSADO:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        data_vin_ult = ultimos_datos["vin_ult_data"]
        data_datamatrix_link = ultimos_datos["datamatrix_link_data"]
        data_datamatrix_num = ultimos_datos["datamatrix_num_data"]
        data_pkn_largo = ultimos_datos["pkn_largo_data"]
        data_cve_com = ultimos_datos["cve_com_data"]
        data_vin_barra = ultimos_datos["vin_barra_data"]

        t_init = time.time()

        data_img_analizada = analizar_imagen(frame, modelo_etiqueta, usar_tracking=True, frame_num=frame_num)

        lista_etiquetas = extraer_todas_las_detecciones(data_img_analizada)

        if not lista_etiquetas:
            if id_etiqueta_actual is not None:
                guardar_etiqueta_en_db(ultimos_datos)
                id_etiqueta_actual = None
                ultimos_datos = {k: "—" for k in ultimos_datos}

            t_fin = time.time() - t_init

            ESTADO_PROCESO["frame"] = optimizar_y_convertir_base64(frame, 640)
            ESTADO_PROCESO["recorte_etiqueta"] = frame_null
            ESTADO_PROCESO["recorte_barcode"] = frame_null
            ESTADO_PROCESO["recorte_qr"] = frame_null
            ESTADO_PROCESO["recorte_vin"] = frame_null
            ESTADO_PROCESO["recorte_datamax"] = frame_null

            ESTADO_PROCESO["vin_ult_data"] = data_vin_ult
            ESTADO_PROCESO["datamatrix_link_data"] = data_datamatrix_link
            ESTADO_PROCESO["datamatrix_num_data"] = data_datamatrix_num
            ESTADO_PROCESO["pkn_largo_data"] = data_pkn_largo
            ESTADO_PROCESO["cve_com_data"] = data_cve_com
            ESTADO_PROCESO["vin_barra_data"] = data_vin_barra

            ESTADO_PROCESO["data_tiempo_procesamiento"] = f"{round(t_fin * 1000, 1)} ms"
            ESTADO_PROCESO["data_num_frame"] = int(frame_num)
            ESTADO_PROCESO["data_num_frame_max"] = int(total_frames)
            continue

        etiqueta_encontrada = max(lista_etiquetas, key=lambda x: x['conf'])

        frame_etiqueta = dibujar_boxes(frame, data_img_analizada)                     # Se dibuja la caja de la etiqueta en la imagen completa
        img_etiqueta = recortar_frame(frame, etiqueta_encontrada['bbox']) 
        
        if img_etiqueta.size == 0:
            continue

        data_codigos_analizados = analizar_imagen(img_etiqueta, modelo_codigos)

        vin_total_det = obtener_mejor_deteccion(data_codigos_analizados, 'VIN_TOTAL')
        vin_ult_det =  obtener_mejor_deteccion(data_codigos_analizados, 'VIN_ULT')
        datamatrix_link_det = obtener_mejor_deteccion(data_codigos_analizados, 'DATAMATRIX_LINK')
        datamatrix_num_det = obtener_mejor_deteccion(data_codigos_analizados, 'DATAMATRIX_NUM')
        pkn_largo_det = obtener_mejor_deteccion(data_codigos_analizados, 'PKN_LARGO')
        cve_com_det = obtener_mejor_deteccion(data_codigos_analizados, 'CVE_COM')
        vin_barra_det = obtener_mejor_deteccion(data_codigos_analizados, 'VIN_BARRA')

        
        img_vin_total_b64, img_vin_ult_b64, img_datamatrix_link_b64 = None, None, None
        img_datamatrix_num_b64, img_pkn_largo_b64, img_cve_com_b64, img_vin_barra_b64 = None, None, None, None

        if vin_total_det:
            crop = recortar_frame(img_etiqueta, vin_total_det['bbox'])
            img_vin_total_b64 = optimizar_y_convertir_base64(crop, 280)


        if vin_ult_det and (frame_num % 5 == 0 or ultimos_datos["vin_ult_data"] == "—"):
            try:
                crop_vin_ult = recortar_frame(img_etiqueta, vin_ult_det['bbox'])
                img_vin_ult_b64 = optimizar_y_convertir_base64(crop_vin_ult, 280)
                scan_vin_ult = MODEL.predict(input=crop_vin_ult)
                if scan_vin_ult:
                    for res in scan_vin_ult:
                        if isinstance(res, dict) and "rec_text" in res and res["rec_text"]:
                            scan_vin_ult_txt = res["rec_text"].strip().upper()
                            if scan_vin_ult_txt:
                                data_vin_ult = scan_vin_ult_txt
                                ultimos_datos["vin_ult_data"] = scan_vin_ult_txt
            except Exception as e:
                logger.error(f"Error en PaddleOCR: {e}")

        if datamatrix_link_det:
            crop_datamatrix_link = recortar_frame(img_etiqueta, datamatrix_link_det['bbox'])
            crop_datamatrix_link, _, _ = rectificar(crop_datamatrix_link)
            img_datamatrix_link_b64 = optimizar_y_convertir_base64(crop_datamatrix_link, 280)
            scan_datamatrix_link = zxingcpp.read_barcode(crop_datamatrix_link)
            if scan_datamatrix_link and scan_datamatrix_link.text: 
                data_datamatrix_link = scan_datamatrix_link.text
                ultimos_datos["datamatrix_link_data"] = scan_datamatrix_link.text


        if datamatrix_num_det:
            crop_datamatrix_num = recortar_frame(img_etiqueta, datamatrix_num_det['bbox'])
            crop_datamatrix_num, _, _ = rectificar(crop_datamatrix_num)
            img_datamatrix_num_b64 = optimizar_y_convertir_base64(crop_datamatrix_num, 280)
            scan_datamatrix_num = zxingcpp.read_barcode(crop_datamatrix_num)
            if scan_datamatrix_num and scan_datamatrix_num.text: 
                data_datamatrix_num = scan_datamatrix_num.text
                ultimos_datos["datamatrix_num_data"] = scan_datamatrix_num.text


        if pkn_largo_det:
            crop_pkn_largo = recortar_frame(img_etiqueta, pkn_largo_det['bbox'])
            crop_pkn_largo, _, _ = rectificar(crop_pkn_largo)
            img_pkn_largo_b64 = optimizar_y_convertir_base64(crop_pkn_largo, 280)
            scan_pkn_largo = zxingcpp.read_barcode(crop_pkn_largo)
            if scan_pkn_largo and scan_pkn_largo.text: 
                data_pkn_largo = scan_pkn_largo.text
                ultimos_datos["pkn_largo_data"] = scan_pkn_largo.text

        if cve_com_det:
            crop_cve_com = recortar_frame(img_etiqueta, cve_com_det['bbox'])
            crop_cve_com, _, _ = rectificar(crop_cve_com)
            img_cve_com_b64 = optimizar_y_convertir_base64(crop_cve_com, 280)
            scan_cve_com = zxingcpp.read_barcode(crop_cve_com)
            if scan_cve_com and scan_cve_com.text: 
                data_cve_com = scan_cve_com.text
                ultimos_datos["cve_com_data"] = scan_cve_com.text

        if vin_barra_det:
            crop_vin_barra = recortar_frame(img_etiqueta, vin_barra_det['bbox'])
            crop_vin_barra, _, _ = rectificar(crop_vin_barra)
            img_vin_barra_b64 = optimizar_y_convertir_base64(crop_vin_barra, 280)
            scan_vin_barra = zxingcpp.read_barcode(crop_vin_barra)
            if scan_vin_barra and scan_vin_barra.text: 
                data_vin_barra = scan_vin_barra.text
                ultimos_datos["vin_barra_data"] = scan_vin_barra.text

        t_fin = time.time() - t_init

        ESTADO_PROCESO["frame"] = optimizar_y_convertir_base64(frame, 640)
        ESTADO_PROCESO["recorte_etiqueta"] = optimizar_y_convertir_base64(frame_etiqueta, 280)

        ESTADO_PROCESO["recorte_vin_total"] = img_vin_total_b64
        ESTADO_PROCESO["recorte_vin_ult"] = img_vin_ult_b64
        ESTADO_PROCESO["recorte_datamatrix_link"] = img_datamatrix_link_b64
        ESTADO_PROCESO["recorte_datamatrix_num"] = img_datamatrix_num_b64
        ESTADO_PROCESO["recorte_pkn_largo"] = img_pkn_largo_b64
        ESTADO_PROCESO["recorte_cve_com"] = img_cve_com_b64
        ESTADO_PROCESO["recorte_vin_barra"] = img_vin_barra_b64

        ESTADO_PROCESO["vin_ult_data"] = data_vin_ult
        ESTADO_PROCESO["datamatrix_link_data"] = data_datamatrix_link
        ESTADO_PROCESO["datamatrix_num_data"] = data_datamatrix_num
        ESTADO_PROCESO["pkn_largo_data"] = data_pkn_largo
        ESTADO_PROCESO["cve_com_data"] = data_cve_com
        ESTADO_PROCESO["vin_barra_data"] = data_vin_barra


        ESTADO_PROCESO["data_tiempo_procesamiento"] = f"{round(t_fin * 1000, 1)} ms"
        ESTADO_PROCESO["data_num_frame"] = int(frame_num)
        ESTADO_PROCESO["data_num_frame_max"] = int(total_frames)

        guardar_etiqueta_en_db(ultimos_datos)
        

    cap.release()
    ESTADO_PROCESO["corriendo"] = False
    frame_num = 0
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
def parar_analisis(payload: VideoPause):
    global DEBE_PARAR
    global ESTA_PAUSADO
    if not ESTADO_PROCESO["corriendo"]:
        return {"status": "error", "message": "No hay ningún análisis activo que detener."}
    
    if payload.pausa_video == 0:
        ESTA_PAUSADO = True

    if payload.pausa_video == 1:
        ESTA_PAUSADO = False
    return {"status": "success", "message": "Se ha enviado la señal de parada al analizador."}

@app.get("/estado")
def obtener_estado():
    """Endpoint al que el Dashboard llamará repetidamente para refrescar su pantalla"""
    return ESTADO_PROCESO

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)