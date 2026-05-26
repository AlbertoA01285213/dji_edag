"""
debug_rtmp.py — Debug en tiempo real via RTMP del DJI Mini 5 Pro.

CONFIGURACION DEL SERVIDOR RTMP (Windows):
  Opcion A — mediamtx (recomendado, sin instalacion):
    1. Descargar mediamtx desde https://github.com/bluenviron/mediamtx/releases
    2. Ejecutar:  mediamtx.exe
    3. URL para DJI Fly: rtmp://<IP_DE_TU_PC>:1935/live/drone

  Opcion B — nginx-rtmp:
    1. Descargar nginx-rtmp-win32 y configurar rtmp { server { listen 1935; ... } }

  En DJI Fly app:
    Perfil → Live → Custom RTMP → rtmp://<IP_DE_TU_PC>:1935/live/drone
    (tu IP local: corre 'ipconfig' en cmd para encontrarla)

Uso:
  python debug_rtmp.py                              # URL por defecto
  python debug_rtmp.py rtmp://192.168.1.X:1935/live/drone
  python debug_rtmp.py rtmp://0.0.0.0:1935/live/drone

Controles:
  ESPACIO  pausar / reanudar
  q        salir
  s        guardar frame actual en output/rtmp_XXXXXX.jpg
  r        reiniciar todos los tracks acumulados
  + / -    ajustar confianza de deteccion de etiqueta (0.05)
"""

import cv2
import os
import re
import sys
import time
import threading
import numpy as np
import zxingcpp
from ultralytics import YOLO

# ── rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODELO_ETIQ    = os.path.join(BASE_DIR, "weights", "modelo_etiqueta.pt")
MODELO_CODIGOS = os.path.join(BASE_DIR, "weights", "modelo_codigos.pt")
OUTPUT_DIR     = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RTMP_URL = "rtmp://0.0.0.0:1935/live/drone"
if len(sys.argv) > 1:
    RTMP_URL = sys.argv[1]

# ── parametros ─────────────────────────────────────────────────────────────────
CONF_ETIQ    = 0.4
CONF_CODIGOS = 0.3
MARGEN_ETIQ  = 10
MARGEN_ZONA  = 8
ZOOM_ANCHO   = 1280
IMGSZ        = 1280
SKIP         = 1      # procesar 1 de cada N frames recibidos (1 = todos)

CAMPOS = ("VIN", "PKN_Largo", "CVE_COM", "DATAMATRIX_NUM", "DATAMATRIX_LINK")
CLASES = {0: "VIN", 1: "PKN_Largo", 2: "CVE_COM", 3: "DATAMATRIX_NUM", 4: "DATAMATRIX_LINK"}

# Colores por tipo de zona (BGR)
COLORES_ZONA = {
    "VIN":             (  0, 140, 255),  # naranja
    "PKN_Largo":       (  0, 220,   0),  # verde
    "CVE_COM":         (  0, 220, 220),  # amarillo
    "DATAMATRIX_NUM":  (220,   0, 220),  # magenta
    "DATAMATRIX_LINK": (220, 220,   0),  # cian
}

KERNEL_SHARP = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], dtype=np.float32)

PATRON_VIN = re.compile(r'^[A-HJ-NPR-Z0-9]{17}$')
PATRON_PKN = re.compile(r'^\d{13,15}$')
PATRON_CVE = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{2}$')
PATRON_DM  = re.compile(r'^[\x20-\x7E]{4,}$')

SEP  = "=" * 62
SEP2 = "-" * 62


# ── lector RTMP en hilo separado ───────────────────────────────────────────────
class RTMPReader:
    """Lee frames RTMP en background y expone siempre el mas reciente."""

    def __init__(self, url, reconectar=True):
        self.url         = url
        self.reconectar  = reconectar
        self._lock       = threading.Lock()
        self._frame      = None
        self._conectado  = False
        self._detenido   = False
        self._thread     = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._detenido:
            print(f"[RTMP] Conectando a {self.url} ...")
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print("[RTMP] No se pudo abrir el stream. Reintentando en 3s...")
                time.sleep(3)
                continue

            print("[RTMP] Conexion establecida.")
            with self._lock:
                self._conectado = True

            while not self._detenido:
                ret, frame = cap.read()
                if not ret:
                    print("[RTMP] Stream interrumpido.")
                    break
                with self._lock:
                    self._frame = frame

            cap.release()
            with self._lock:
                self._conectado = False

            if not self.reconectar or self._detenido:
                break
            print("[RTMP] Reconectando en 2s...")
            time.sleep(2)

    def leer(self):
        """Devuelve (conectado, frame_o_None)."""
        with self._lock:
            return self._conectado, (self._frame.copy() if self._frame is not None else None)

    def detener(self):
        self._detenido = True


# ── helpers de imagen ──────────────────────────────────────────────────────────
def nitidez(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def recortar(img, x1, y1, x2, y2, margen=0):
    h, w = img.shape[:2]
    return img[max(0, y1-margen):min(h, y2+margen),
               max(0, x1-margen):min(w, x2+margen)]


def zoom(img, target_w=ZOOM_ANCHO):
    if img.shape[1] >= target_w:
        return img
    f = target_w / img.shape[1]
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)


def escalar(img, target_w):
    if img.shape[1] < target_w:
        f = target_w / img.shape[1]
        return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return img


def variantes(gris):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gc    = clahe.apply(gris)
    ns    = cv2.filter2D(gc, -1, KERNEL_SHARP)
    return [
        gris, gc, ns,
        cv2.adaptiveThreshold(gc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 21, 6),
    ]


# ── lectura de codigos ─────────────────────────────────────────────────────────
def leer_1d(crop_bgr, target_w=2400):
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    g = escalar(g, target_w)
    resultados = set()
    for v in variantes(g):
        for c in zxingcpp.read_barcodes(v, try_rotate=True):
            t = c.text.strip()
            if t:
                resultados.add(t)
    return resultados


def leer_dm(crop_bgr, target_w=600):
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    g = escalar(g, target_w)
    for v in variantes(g):
        for c in zxingcpp.read_barcodes(
                v, formats=zxingcpp.BarcodeFormat.DataMatrix, try_rotate=True):
            t = c.text.strip()
            if t and PATRON_DM.match(t):
                return t
    return None


# ── deteccion de zonas ─────────────────────────────────────────────────────────
def detectar_zonas(etiqueta, model_codigos):
    res   = model_codigos(etiqueta, conf=CONF_CODIGOS, verbose=False)[0]
    zonas = {v: [] for v in CLASES.values()}

    for box in res.boxes:
        cid = int(box.cls[0])
        if cid not in CLASES:
            continue
        nombre = CLASES[cid]
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = recortar(etiqueta, x1, y1, x2, y2, MARGEN_ZONA)
        if crop.size == 0:
            continue
        zonas[nombre].append({"crop": crop, "conf": conf, "bbox": (x1, y1, x2, y2)})

    for nombre in ("PKN_Largo", "CVE_COM", "DATAMATRIX_NUM", "DATAMATRIX_LINK"):
        if len(zonas[nombre]) > 1:
            zonas[nombre] = [max(zonas[nombre], key=lambda d: d["conf"])]

    zonas["VIN"] = sorted(zonas["VIN"], key=lambda d: d["conf"], reverse=True)[:3]
    return zonas


def analizar_pendientes(crop_etiq, model_codigos, pendientes):
    etiqueta    = zoom(crop_etiq)
    zonas       = detectar_zonas(etiqueta, model_codigos)
    nuevos      = {}
    detalle_vin = []

    if "VIN" in pendientes:
        for i, det in enumerate(zonas["VIN"], start=1):
            textos = leer_1d(det["crop"])
            vin_ok = next((t for t in textos if PATRON_VIN.match(t)), None)
            detalle_vin.append({"intento": i, "conf": det["conf"],
                                "bbox": det["bbox"], "textos": textos, "valido": vin_ok})
            if vin_ok and "VIN" not in nuevos:
                nuevos["VIN"] = vin_ok

    if "PKN_Largo" in pendientes:
        for det in zonas["PKN_Largo"]:
            textos = leer_1d(det["crop"])
            pkn = next((t for t in textos if PATRON_PKN.match(t)), None)
            if pkn:
                nuevos["PKN_Largo"] = pkn
                break

    if "CVE_COM" in pendientes:
        for det in zonas["CVE_COM"]:
            textos = leer_1d(det["crop"])
            cve = next((t for t in textos if PATRON_CVE.match(t)), None)
            if cve:
                nuevos["CVE_COM"] = cve
                break

    if "DATAMATRIX_NUM" in pendientes:
        for det in zonas["DATAMATRIX_NUM"]:
            v = leer_dm(det["crop"])
            if v:
                nuevos["DATAMATRIX_NUM"] = v
                break

    if "DATAMATRIX_LINK" in pendientes:
        for det in zonas["DATAMATRIX_LINK"]:
            v = leer_dm(det["crop"])
            if v:
                nuevos["DATAMATRIX_LINK"] = v
                break

    return nuevos, detalle_vin, zonas


# ── visualizacion ──────────────────────────────────────────────────────────────
ETIQUETAS_CORTAS = {
    "VIN":             "VIN",
    "PKN_Largo":       "PKN",
    "CVE_COM":         "CVE",
    "DATAMATRIX_NUM":  "DM NUM",
    "DATAMATRIX_LINK": "DM LINK",
}


def _panel_campos(vis, tracks, boxes, conf_umbral):
    """Dibuja panel semitransparente con los 5 campos del track mas confiable."""
    candidatos = [b for b in boxes
                  if b.id is not None and float(b.conf[0]) >= conf_umbral]
    if not candidatos:
        return

    mejor = max(candidatos, key=lambda b: float(b.conf[0]))
    track_id = int(mejor.id[0])
    estado   = tracks.get(track_id, {})
    leidos   = sum(1 for v in estado.values() if v)

    h, w = vis.shape[:2]
    LINE_H   = 34
    PAD      = 12
    panel_w  = 420
    panel_h  = PAD + 30 + len(CAMPOS) * LINE_H + PAD
    px       = w - panel_w - 14
    py       = 44

    # Fondo semitransparente
    roi = vis[py: py + panel_h, px: px + panel_w]
    fondo = np.zeros_like(roi)
    cv2.addWeighted(fondo, 0.55, roi, 0.45, 0, roi)
    vis[py: py + panel_h, px: px + panel_w] = roi

    # Borde
    cv2.rectangle(vis, (px, py), (px + panel_w, py + panel_h), (180, 180, 180), 1)

    # Encabezado
    header = f"Track ID:{track_id}  [{leidos}/{len(CAMPOS)}]"
    color_h = (0, 255, 0) if leidos == len(CAMPOS) else (0, 200, 255)
    cv2.putText(vis, header, (px + PAD, py + PAD + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_h, 1, cv2.LINE_AA)

    # Separador
    sy = py + PAD + 26
    cv2.line(vis, (px + PAD, sy), (px + panel_w - PAD, sy), (100, 100, 100), 1)

    # Filas de campos
    for i, campo in enumerate(CAMPOS):
        val   = estado.get(campo)
        cy    = sy + 6 + i * LINE_H
        color = (0, 220, 0) if val else (90, 90, 90)

        # Indicador izquierdo
        dot_color = (0, 220, 0) if val else (60, 60, 200)
        cv2.circle(vis, (px + PAD + 6, cy + 10), 5, dot_color, -1)

        # Nombre corto
        etiq = ETIQUETAS_CORTAS[campo]
        cv2.putText(vis, etiq, (px + PAD + 18, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Valor leido (truncado si es muy largo)
        valor = val if val else "---"
        if len(valor) > 34:
            valor = valor[:31] + "..."
        cv2.putText(vis, valor, (px + PAD + 90, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def dibujar_frame(frame, boxes, conf_umbral, frame_num, fps_proc, conectado, tracks):
    vis = frame.copy()

    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_umbral:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        nit  = nitidez(crop) if crop.size > 0 else 0

        estado   = tracks.get(track_id, {})
        leidos   = sum(1 for v in estado.values() if v)
        completo = leidos == len(CAMPOS)

        color = (0, 255, 0) if completo else (0, int(200 * conf), int(200 * (1 - conf)))
        grosor = 3 if completo else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, grosor)

        label = f"ID:{track_id} {conf:.2f} nit:{nit:.0f} [{leidos}/{len(CAMPOS)}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        by = max(y1 - 4, th + 4)
        cv2.rectangle(vis, (x1, by - th - 4), (x1 + tw + 4, by + 2), color, -1)
        cv2.putText(vis, label, (x1 + 2, by - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # Panel de campos
    _panel_campos(vis, tracks, boxes, conf_umbral)

    # HUD inferior
    estado_rtmp = "CONECTADO" if conectado else "SIN STREAM"
    color_hud   = (0, 255, 0) if conectado else (0, 0, 255)
    hud = (f"Frame {frame_num} | {estado_rtmp} | "
           f"conf>={conf_umbral:.2f} | {fps_proc:.1f} fps proc")
    cv2.putText(vis, hud, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_hud, 2, cv2.LINE_AA)
    return vis


def dibujar_crop_con_zonas(crop_etiq, zonas, estado):
    """Muestra el crop de etiqueta con las zonas detectadas y estado de lectura."""
    etiq = zoom(crop_etiq.copy())
    target_w = 900
    h, w = etiq.shape[:2]
    if w > target_w:
        scale = target_w / w
        etiq  = cv2.resize(etiq, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        h, w  = etiq.shape[:2]

    # Escalar coordenadas de zonas al mismo factor
    scale_x = w / zoom(crop_etiq).shape[1]
    scale_y = h / zoom(crop_etiq).shape[0]

    for campo, dets in zonas.items():
        color = COLORES_ZONA.get(campo, (200, 200, 200))
        val   = estado.get(campo)
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            sx1 = int(x1 * scale_x); sy1 = int(y1 * scale_y)
            sx2 = int(x2 * scale_x); sy2 = int(y2 * scale_y)
            cv2.rectangle(etiq, (sx1, sy1), (sx2, sy2), color, 2)

            # Texto encima del recuadro
            texto = val if val else f"{campo} {det['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(sy1 - 4, th + 4)
            cv2.rectangle(etiq, (sx1, ty - th - 3), (sx1 + tw + 4, ty + 1), color, -1)
            cv2.putText(etiq, texto, (sx1 + 2, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Panel de estado en la parte inferior
    panel_h = len(CAMPOS) * 22 + 10
    panel   = np.zeros((panel_h, w, 3), dtype=np.uint8)
    for i, campo in enumerate(CAMPOS):
        val   = estado.get(campo)
        color = (0, 220, 0) if val else (60, 60, 60)
        marca = "OK" if val else "  "
        texto = f"[{marca}] {campo:<18}  {val or '---'}"
        cv2.putText(panel, texto, (8, 18 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    return np.vstack([etiq, panel])


def dibujar_sin_deteccion():
    blank = np.zeros((300, 900, 3), dtype=np.uint8)
    cv2.putText(blank, "Sin etiqueta detectada", (220, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2)
    return blank


# ── consola ────────────────────────────────────────────────────────────────────
def log_nuevo_track(track_id, frame_num):
    print(f"\n{SEP2}")
    print(f"  NUEVO TRACK  ID:{track_id}  frame {frame_num}")
    print(SEP2)


def log_actualizacion(frame_num, track_id, estado, nuevos, detalle_vin, t_cod):
    leidos = sum(1 for v in estado.values() if v)
    print(f"\n  f{frame_num:>6}  ID:{track_id}  [{leidos}/{len(CAMPOS)}]  "
          f"+{len(nuevos)} nuevos  ({t_cod:.2f}s)")
    for campo in CAMPOS:
        val      = estado.get(campo)
        es_nuevo = campo in nuevos
        marca    = "OK+" if es_nuevo else ("OK " if val else "   ")
        print(f"    [{marca}] {campo:<18}  {val or '(sin lectura)'}")

    if "VIN" in nuevos or detalle_vin:
        print("    VIN — intentos:")
        for d in detalle_vin:
            print(f"      Intento {d['intento']}  conf={d['conf']:.2f}  bbox={d['bbox']}")
            for t in sorted(d["textos"]):
                marca = "  <- VALIDO" if PATRON_VIN.match(t) else ""
                print(f"        leido: {t}{marca}")


def log_completo(track_id, estado, frame_num):
    print(f"\n{SEP}")
    print(f"  *** COMPLETO  Track ID:{track_id}  en frame {frame_num} ***")
    for campo in CAMPOS:
        print(f"    {campo:<18}  {estado[campo]}")
    print(SEP)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\nCargando modelos...")
    model_etiq    = YOLO(MODELO_ETIQ)
    model_codigos = YOLO(MODELO_CODIGOS)
    print(f"Modelos cargados.")
    print(f"\nURL RTMP: {RTMP_URL}")
    print("Controles: ESPACIO=pausa  q=salir  s=guardar  r=reset tracks  +/- confianza")
    print(SEP)

    reader = RTMPReader(RTMP_URL)

    conf       = CONF_ETIQ
    pausado    = False
    frame_num  = 0
    fps_proc   = 0.0
    last_boxes = []
    tracks     = {}      # track_id -> dict campo -> valor
    completados = set()
    ultima_vis  = None
    ultima_zona_img = None

    # Estado de la ultima zona detectada (para la ventana de crop)
    ultimo_crop  = None
    ultimo_zonas = {}
    ultimo_track = None

    try:
        while True:
            conectado, frame = reader.leer()

            if frame is None:
                # Mostrar pantalla de espera
                espera = np.zeros((400, 720, 3), dtype=np.uint8)
                msg    = "Esperando stream RTMP..." if not conectado else "Procesando..."
                cv2.putText(espera, msg, (150, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 255), 2)
                cv2.putText(espera, RTMP_URL, (40, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
                cv2.imshow("Debug RTMP — DJI Mini 5 Pro", espera)
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
                continue

            if pausado:
                if ultima_vis is not None:
                    cv2.imshow("Debug RTMP — DJI Mini 5 Pro", ultima_vis)
                if ultima_zona_img is not None:
                    cv2.imshow("Crop + Zonas", ultima_zona_img)
                key = cv2.waitKey(30) & 0xFF
                if key == ord(' '):
                    pausado = False
                    print(f"REANUDADO en frame {frame_num}")
                elif key == ord('q'):
                    break
                elif key == ord('s') and ultima_vis is not None:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(OUTPUT_DIR, f"rtmp_{ts}.jpg")
                    cv2.imwrite(path, ultima_vis)
                    print(f"  [GUARDADO] {path}")
                continue

            frame_num += 1
            if frame_num % SKIP != 0:
                if ultima_vis is not None:
                    cv2.imshow("Debug RTMP — DJI Mini 5 Pro", ultima_vis)
                cv2.waitKey(1)
                continue

            # ── deteccion de etiquetas ─────────────────────────────────────────
            t_ini = time.time()
            res   = model_etiq.track(frame, conf=conf, imgsz=IMGSZ,
                                     persist=True, tracker="bytetrack.yaml",
                                     verbose=False)[0]
            dt       = time.time() - t_ini
            fps_proc = 1.0 / dt if dt > 0 else 0.0

            last_boxes = res.boxes if res.boxes.id is not None else []

            # ── por cada track: leer codigos si no esta completo ───────────────
            mejor_crop_nit  = -1
            mejor_crop_data = None  # (crop, track_id, zonas, estado)

            for box in last_boxes:
                if box.id is None:
                    continue
                track_id = int(box.id[0])
                conf_box = float(box.conf[0])
                if conf_box < conf:
                    continue

                if track_id not in tracks:
                    tracks[track_id] = {k: None for k in CAMPOS}
                    log_nuevo_track(track_id, frame_num)

                estado    = tracks[track_id]
                pendientes = {k for k, v in estado.items() if v is None}

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop_etiq = recortar(frame, x1, y1, x2, y2, MARGEN_ETIQ)
                if crop_etiq.size == 0:
                    continue

                nit = nitidez(crop_etiq)

                if track_id not in completados:
                    t0 = time.time()
                    nuevos, detalle_vin, zonas_det = analizar_pendientes(
                        crop_etiq, model_codigos, pendientes)
                    t_cod = time.time() - t0

                    if nuevos or detalle_vin:
                        for k, v in nuevos.items():
                            estado[k] = v
                        log_actualizacion(frame_num, track_id, estado,
                                          nuevos, detalle_vin, t_cod)
                        if all(v is not None for v in estado.values()):
                            log_completo(track_id, estado, frame_num)
                            completados.add(track_id)
                else:
                    zonas_det = detectar_zonas(zoom(crop_etiq), model_codigos)

                # Conservar el crop mas nitido para la ventana de visualizacion
                if nit > mejor_crop_nit:
                    mejor_crop_nit  = nit
                    mejor_crop_data = (crop_etiq, track_id, zonas_det, estado.copy())

            # ── dibujar ────────────────────────────────────────────────────────
            ultima_vis = dibujar_frame(frame, last_boxes, conf,
                                       frame_num, fps_proc, conectado, tracks)
            cv2.imshow("Debug RTMP — DJI Mini 5 Pro", ultima_vis)

            if mejor_crop_data is not None:
                crop_etiq, tid, zonas_det, estado = mejor_crop_data
                ultima_zona_img = dibujar_crop_con_zonas(crop_etiq, zonas_det, estado)
            else:
                ultima_zona_img = dibujar_sin_deteccion()

            cv2.imshow("Crop + Zonas", ultima_zona_img)

            # ── teclado ────────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Saliendo.")
                break
            elif key == ord(' '):
                pausado = not pausado
                print(f"{'PAUSADO' if pausado else 'REANUDADO'} en frame {frame_num}")
            elif key == ord('s'):
                ts   = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(OUTPUT_DIR, f"rtmp_{ts}.jpg")
                cv2.imwrite(path, ultima_vis)
                print(f"  [GUARDADO] {path}")
            elif key == ord('r'):
                tracks.clear()
                completados.clear()
                print(f"\n[RESET] Todos los tracks reiniciados en frame {frame_num}")
                print(SEP2)
            elif key in (ord('+'), ord('=')):
                conf = min(0.95, round(conf + 0.05, 2))
                print(f"  Confianza -> {conf:.2f}")
            elif key == ord('-'):
                conf = max(0.05, round(conf - 0.05, 2))
                print(f"  Confianza -> {conf:.2f}")

    except KeyboardInterrupt:
        print("\n[Interrumpido]")
    finally:
        reader.detener()
        cv2.destroyAllWindows()

        print(f"\n{'#'*62}")
        print(f"  FIN  —  {frame_num} frames procesados")
        print(f"  Tracks completos  : {len(completados)}")
        print(f"  Tracks incompletos: {len(tracks) - len(completados)}")
        for tid, estado in sorted(tracks.items()):
            leidos = sum(1 for v in estado.values() if v)
            marca  = "COMPLETO" if tid in completados else f"{leidos}/{len(CAMPOS)}"
            print(f"  ID:{tid:<4}  {marca}")
            for campo in CAMPOS:
                print(f"    {campo:<18}  {estado.get(campo) or '(sin lectura)'}")
        print(f"{'#'*62}")


if __name__ == "__main__":
    main()
