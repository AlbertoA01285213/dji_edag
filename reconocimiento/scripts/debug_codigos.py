"""
debug_codigos.py — Deteccion de zonas + lectura zxingcpp con visualizacion y video de salida.

Flujo:
  1. modelo_etiqueta detecta etiquetas en el frame (bytetrack)
  2. Por cada track activo se recorta y escala la etiqueta
  3. modelo_codigos detecta las 5 zonas
  4. Para VIN se intentan TODOS los crops de clase VIN (hasta 3); gana el primero valido
  5. Solo se intenta leer los campos que aun no se han leido para ese track
  6. Una vez que un track tiene los 5 campos, se ignora para siempre

Salida:
  - Consola con detalle de lecturas
  - Ventana en tiempo real con bounding boxes y panel de campos
  - Video MP4 en output/ con todos los frames anotados

Controles:
  ESPACIO  pausar / reanudar
  q        salir (guarda el video hasta ese punto)
  s        guardar screenshot del frame actual
"""

import cv2
import os
import re
import sys
import time
import numpy as np
import zxingcpp
from ultralytics import YOLO

# ── rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODELO_ETIQ    = os.path.join(BASE_DIR, "weights", "modelo_etiqueta.pt")
MODELO_CODIGOS = os.path.join(BASE_DIR, "weights", "modelo_codigos.pt")
OUTPUT_DIR     = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO = os.path.join(BASE_DIR, "input", "DJI_20260525151408_0197_D.MP4")
if len(sys.argv) > 1:
    VIDEO = sys.argv[1]

# ── parametros ────────────────────────────────────────────────────────────────
SKIP         = 2
CONF_ETIQ    = 0.4
CONF_CODIGOS = 0.3
MARGEN_ETIQ  = 10
MARGEN_ZONA  = 8
ZOOM_ANCHO   = 1280
GRABAR       = True   # generar video de salida en output/

CAMPOS = ("VIN", "PKN_Largo", "CVE_COM", "DATAMATRIX_NUM", "DATAMATRIX_LINK")
CLASES = {0: "VIN", 1: "PKN_Largo", 2: "CVE_COM", 3: "DATAMATRIX_NUM", 4: "DATAMATRIX_LINK"}

ETIQUETAS_CORTAS = {
    "VIN": "VIN", "PKN_Largo": "PKN", "CVE_COM": "CVE",
    "DATAMATRIX_NUM": "DM NUM", "DATAMATRIX_LINK": "DM LINK",
}
COLORES_ZONA = {
    "VIN":             (  0, 140, 255),
    "PKN_Largo":       (  0, 220,   0),
    "CVE_COM":         (  0, 220, 220),
    "DATAMATRIX_NUM":  (220,   0, 220),
    "DATAMATRIX_LINK": (220, 220,   0),
}

# ── patrones ─────────────────────────────────────────────────────────────────
PATRON_VIN = re.compile(r'^[A-HJ-NPR-Z0-9]{17}$')
PATRON_PKN = re.compile(r'^\d{13,15}$')
PATRON_CVE = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{2}$')
PATRON_DM  = re.compile(r'^[\x20-\x7E]{4,}$')

KERNEL_SHARP = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], dtype=np.float32)
PADDING_1D   = 40   # zona silenciosa requerida por Code39

SEP  = "=" * 62
SEP2 = "-" * 62


# ── helpers de imagen ─────────────────────────────────────────────────────────
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
    _, otsu = cv2.threshold(gc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [
        gris, gc, ns, otsu,
        cv2.adaptiveThreshold(gc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 21, 6),
    ]


def con_padding(img, pad=PADDING_1D):
    """Agrega margen blanco — zona silenciosa requerida por Code39."""
    return cv2.copyMakeBorder(img, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=255)


# ── rectificacion de perspectiva ──────────────────────────────────────────────
def _ordenar_puntos(pts):
    pts  = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s       = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d       = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def rectificar_etiqueta(crop_bgr):
    """
    Detecta el contorno de la etiqueta por contraste y corrige la perspectiva.
    Devuelve (imagen_rectificada, exito, contorno_pts).
    contorno_pts: los 4 puntos encontrados (para dibujar en debug), o None.
    """
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    edges  = cv2.Canny(blur, 25, 90)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges  = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop_bgr, False, None

    area_min   = h * w * 0.15
    candidatos = [c for c in contours if cv2.contourArea(c) > area_min]
    if not candidatos:
        return crop_bgr, False, None

    mayor  = max(candidatos, key=cv2.contourArea)
    peri   = cv2.arcLength(mayor, True)
    approx = cv2.approxPolyDP(mayor, 0.02 * peri, True)

    if len(approx) == 4:
        pts = _ordenar_puntos(approx)
    else:
        rect = cv2.minAreaRect(mayor)
        box  = cv2.boxPoints(rect)
        pts  = _ordenar_puntos(box)

    tl, tr, br, bl = pts
    dst_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    dst_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    if dst_w < 50 or dst_h < 50:
        return crop_bgr, False, None

    pts_orig = pts.copy()
    if dst_w > dst_h:
        dst_w, dst_h = dst_h, dst_w
        pts = np.array([tr, br, bl, tl], dtype=np.float32)

    dst = np.array([
        [0,         0        ],
        [dst_w - 1, 0        ],
        [dst_w - 1, dst_h - 1],
        [0,         dst_h - 1],
    ], dtype=np.float32)

    M      = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(crop_bgr, M, (dst_w, dst_h))
    return warped, True, pts_orig


# ── lectura de codigos ────────────────────────────────────────────────────────
_BINARIZERS_1D = [
    zxingcpp.Binarizer.LocalAverage,
    zxingcpp.Binarizer.GlobalHistogram,
    zxingcpp.Binarizer.FixedThreshold,
]


def leer_1d(crop_bgr, target_w=2400):
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    g = escalar(g, target_w)
    resultados = set()
    for v in variantes(g):
        vp = con_padding(v)
        for binarizer in _BINARIZERS_1D:
            for c in zxingcpp.read_barcodes(
                    vp, try_rotate=True, try_invert=True, binarizer=binarizer):
                t = c.text.strip()
                if t:
                    resultados.add(t)
            for c in zxingcpp.read_barcodes(
                    vp, formats=zxingcpp.BarcodeFormat.Code39,
                    try_rotate=True, try_invert=True, binarizer=binarizer):
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


# ── deteccion de zonas ────────────────────────────────────────────────────────
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
    # Rectificar perspectiva antes de escanear
    rectificada, ok, _ = rectificar_etiqueta(crop_etiq)
    etiqueta = zoom(rectificada if ok else crop_etiq)
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


# ── visualizacion ─────────────────────────────────────────────────────────────
def _panel_campos(vis, tracks, boxes, conf_umbral):
    candidatos = [b for b in boxes
                  if b.id is not None and float(b.conf[0]) >= conf_umbral]
    if not candidatos:
        return

    mejor    = max(candidatos, key=lambda b: float(b.conf[0]))
    track_id = int(mejor.id[0])
    estado   = tracks.get(track_id, {})
    leidos   = sum(1 for v in estado.values() if v)

    h, w    = vis.shape[:2]
    LINE_H  = 34
    PAD     = 12
    panel_w = 420
    panel_h = PAD + 30 + len(CAMPOS) * LINE_H + PAD
    px      = w - panel_w - 14
    py      = 44

    roi   = vis[py: py + panel_h, px: px + panel_w]
    fondo = np.zeros_like(roi)
    cv2.addWeighted(fondo, 0.55, roi, 0.45, 0, roi)
    vis[py: py + panel_h, px: px + panel_w] = roi
    cv2.rectangle(vis, (px, py), (px + panel_w, py + panel_h), (180, 180, 180), 1)

    header  = f"Track ID:{track_id}  [{leidos}/{len(CAMPOS)}]"
    color_h = (0, 255, 0) if leidos == len(CAMPOS) else (0, 200, 255)
    cv2.putText(vis, header, (px + PAD, py + PAD + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_h, 1, cv2.LINE_AA)

    sy = py + PAD + 26
    cv2.line(vis, (px + PAD, sy), (px + panel_w - PAD, sy), (100, 100, 100), 1)

    for i, campo in enumerate(CAMPOS):
        val   = estado.get(campo)
        cy    = sy + 6 + i * LINE_H
        color = (0, 220, 0) if val else (90, 90, 90)

        dot_color = (0, 220, 0) if val else (60, 60, 200)
        cv2.circle(vis, (px + PAD + 6, cy + 10), 5, dot_color, -1)

        cv2.putText(vis, ETIQUETAS_CORTAS[campo], (px + PAD + 18, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        valor = (val or "---")
        if len(valor) > 34:
            valor = valor[:31] + "..."
        cv2.putText(vis, valor, (px + PAD + 90, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def dibujar(frame, boxes, conf_umbral, frame_num, fps_proc, tracks):
    vis = frame.copy()

    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_umbral:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        crop     = frame[max(0, y1):y2, max(0, x1):x2]
        nit      = nitidez(crop) if crop.size > 0 else 0

        estado   = tracks.get(track_id, {})
        leidos   = sum(1 for v in estado.values() if v)
        completo = leidos == len(CAMPOS)

        color  = (0, 255, 0) if completo else (0, int(200 * conf), int(200 * (1 - conf)))
        grosor = 3 if completo else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, grosor)

        label = f"ID:{track_id} {conf:.2f} nit:{nit:.0f} [{leidos}/{len(CAMPOS)}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        by = max(y1 - 4, th + 4)
        cv2.rectangle(vis, (x1, by - th - 4), (x1 + tw + 4, by + 2), color, -1)
        cv2.putText(vis, label, (x1 + 2, by - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    _panel_campos(vis, tracks, boxes, conf_umbral)

    hud = f"Frame {frame_num} | conf>={conf_umbral:.2f} | {fps_proc:.1f} fps proc"
    cv2.putText(vis, hud, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def mostrar_crop_zonas(frame, boxes, conf_umbral, tracks, zonas_activas):
    """
    Dos ventanas:
      'Original + contorno' — crop YOLO con el cuadrilatero detectado dibujado
      'Rectificada + Zonas' — etiqueta corregida con zonas y panel de estado
    """
    candidatos = [b for b in boxes
                  if b.id is not None and float(b.conf[0]) >= conf_umbral]
    if not candidatos:
        blank = np.zeros((220, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Sin deteccion", (180, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2)
        cv2.imshow("Original + contorno", blank)
        cv2.imshow("Rectificada + Zonas", blank)
        return

    mejor    = max(candidatos, key=lambda b: nitidez(
        frame[max(0, int(b.xyxy[0][1])):int(b.xyxy[0][3]),
              max(0, int(b.xyxy[0][0])):int(b.xyxy[0][2])]))
    track_id = int(mejor.id[0])
    x1, y1, x2, y2 = map(int, mejor.xyxy[0])
    crop_raw = recortar(frame, x1, y1, x2, y2, MARGEN_ETIQ)
    if crop_raw.size == 0:
        return

    # ── ventana 1: original con contorno detectado ────────────────────────────
    rectificada, ok, pts_contorno = rectificar_etiqueta(crop_raw)

    vis_orig = crop_raw.copy()
    target_w = 900
    ho, wo   = vis_orig.shape[:2]
    if wo > target_w:
        sc_o    = target_w / wo
        vis_orig = cv2.resize(vis_orig, None, fx=sc_o, fy=sc_o,
                              interpolation=cv2.INTER_LINEAR)
        ho, wo  = vis_orig.shape[:2]
        sc_x = wo / crop_raw.shape[1]
        sc_y = ho / crop_raw.shape[0]
    else:
        sc_x = sc_y = 1.0

    if ok and pts_contorno is not None:
        pts_draw = (pts_contorno * np.array([sc_x, sc_y])).astype(np.int32)
        cv2.polylines(vis_orig, [pts_draw], isClosed=True, color=(0, 255, 0), thickness=2)
        for i, pt in enumerate(pts_draw):
            cv2.circle(vis_orig, tuple(pt), 6, (0, 0, 255), -1)
            cv2.putText(vis_orig, str(i), (pt[0]+6, pt[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        etiq_label = "Rectificacion OK"
        color_lbl  = (0, 255, 0)
    else:
        etiq_label = "Sin rectificacion (fallback)"
        color_lbl  = (0, 100, 255)
    cv2.putText(vis_orig, etiq_label, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_lbl, 2, cv2.LINE_AA)
    cv2.imshow("Original + contorno", vis_orig)

    # ── ventana 2: etiqueta rectificada con zonas ─────────────────────────────
    base   = rectificada if ok else crop_raw
    etiq   = zoom(base.copy())
    eh, ew = etiq.shape[:2]
    if ew > target_w:
        sc2  = target_w / ew
        etiq = cv2.resize(etiq, None, fx=sc2, fy=sc2, interpolation=cv2.INTER_LINEAR)
        eh, ew = etiq.shape[:2]

    sx    = ew / zoom(base).shape[1]
    sy_sc = eh / zoom(base).shape[0]

    zonas  = zonas_activas.get(track_id, {})
    estado = tracks.get(track_id, {})

    for campo, dets in zonas.items():
        color = COLORES_ZONA.get(campo, (200, 200, 200))
        val   = estado.get(campo)
        for det in dets:
            bx1, by1, bx2, by2 = det["bbox"]
            rx1 = int(bx1 * sx);  ry1 = int(by1 * sy_sc)
            rx2 = int(bx2 * sx);  ry2 = int(by2 * sy_sc)
            cv2.rectangle(etiq, (rx1, ry1), (rx2, ry2), color, 2)
            texto = val if val else f"{ETIQUETAS_CORTAS[campo]} {det['conf']:.2f}"
            if len(texto) > 30:
                texto = texto[:27] + "..."
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(ry1 - 4, th + 4)
            cv2.rectangle(etiq, (rx1, ty - th - 3), (rx1 + tw + 4, ty + 1), color, -1)
            cv2.putText(etiq, texto, (rx1 + 2, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    panel_h = len(CAMPOS) * 22 + 10
    panel   = np.zeros((panel_h, ew, 3), dtype=np.uint8)
    for i, campo in enumerate(CAMPOS):
        val   = estado.get(campo)
        color = (0, 220, 0) if val else (60, 60, 60)
        texto = f"[{'OK' if val else '  '}] {campo:<18}  {val or '---'}"
        cv2.putText(panel, texto, (8, 18 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    cv2.imshow("Rectificada + Zonas", np.vstack([etiq, panel]))


# ── consola ───────────────────────────────────────────────────────────────────
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


def log_resumen(tracks, completados, total_frames):
    print(f"\n{'#'*62}")
    print(f"  FIN  —  {total_frames} frames procesados")
    print(f"  Tracks completos  : {len(completados)}")
    print(f"  Tracks incompletos: {len(tracks) - len(completados)}")
    for tid, estado in sorted(tracks.items()):
        leidos = sum(1 for v in estado.values() if v)
        marca  = "COMPLETO" if tid in completados else f"{leidos}/{len(CAMPOS)}"
        print(f"  ID:{tid:<4}  {marca}")
        for campo in CAMPOS:
            print(f"    {campo:<18}  {estado.get(campo) or '(sin lectura)'}")
    print(f"{'#'*62}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(VIDEO):
        print(f"[ERROR] Video no encontrado: {VIDEO}")
        print("Uso: python debug_codigos.py [ruta_video]")
        return

    print("Cargando modelos...")
    model_etiq    = YOLO(MODELO_ETIQ)
    model_codigos = YOLO(MODELO_CODIGOS)

    cap       = cv2.VideoCapture(VIDEO)
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    ancho     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nombre    = os.path.basename(VIDEO)
    stem      = os.path.splitext(nombre)[0]

    # Video de salida
    writer = None
    ruta_out = None
    if GRABAR:
        ruta_out = os.path.join(OUTPUT_DIR, f"debug_codigos_{stem}.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(ruta_out, fourcc, fps_video, (ancho, alto))
        print(f"Grabando en: {ruta_out}")

    print(f"Video : {nombre} | {total} frames | {fps_video:.1f} fps | skip={SKIP}")
    print(f"conf_etiq={CONF_ETIQ}  conf_codigos={CONF_CODIGOS}")
    print("Controles: ESPACIO=pausa  q=salir  s=screenshot")
    print(SEP)

    tracks      = {}
    completados = set()
    zonas_activas = {}   # track_id -> zonas del ultimo intento
    frame_num   = 0
    fps_proc    = 0.0
    last_boxes  = []
    last_vis    = None
    pausado     = False

    try:
        while True:
            if pausado:
                if last_vis is not None:
                    cv2.imshow("Debug codigos", last_vis)
                key = cv2.waitKey(30) & 0xFF
                if key == ord(' '):
                    pausado = False
                    print(f"REANUDADO en frame {frame_num}")
                elif key == ord('q'):
                    print("Saliendo.")
                    break
                elif key == ord('s') and last_vis is not None:
                    p = os.path.join(OUTPUT_DIR, f"debug_f{frame_num:06d}.jpg")
                    cv2.imwrite(p, last_vis)
                    print(f"  [SCREENSHOT] {p}")
                continue

            ret, frame = cap.read()
            if not ret:
                print("\nFin del video.")
                break
            frame_num += 1

            if frame_num % SKIP != 0:
                # Frame no procesado: redibujar con estado anterior
                vis = dibujar(frame, last_boxes, CONF_ETIQ, frame_num, fps_proc, tracks)
                if writer:
                    writer.write(vis)
                last_vis = vis
                cv2.imshow("Debug codigos", vis)
                mostrar_crop_zonas(frame, last_boxes, CONF_ETIQ, tracks, zonas_activas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Saliendo.")
                    break
                elif key == ord(' '):
                    pausado = True
                    print(f"PAUSADO en frame {frame_num}")
                elif key == ord('s'):
                    p = os.path.join(OUTPUT_DIR, f"debug_f{frame_num:06d}.jpg")
                    cv2.imwrite(p, vis)
                    print(f"  [SCREENSHOT] {p}")
                continue

            # ── deteccion de etiquetas ────────────────────────────────────────
            t_ini = time.time()
            res   = model_etiq.track(frame, conf=CONF_ETIQ, imgsz=1280,
                                     persist=True, tracker="bytetrack.yaml",
                                     verbose=False)[0]
            dt       = time.time() - t_ini
            fps_proc = 1.0 / dt if dt > 0 else 0.0

            last_boxes = res.boxes if res.boxes.id is not None else []

            # ── lectura de codigos por track ──────────────────────────────────
            for box in last_boxes:
                if box.id is None:
                    continue
                track_id = int(box.id[0])
                if track_id not in tracks:
                    tracks[track_id] = {k: None for k in CAMPOS}
                    log_nuevo_track(track_id, frame_num)

                if track_id in completados:
                    continue

                estado     = tracks[track_id]
                pendientes = {k for k, v in estado.items() if v is None}

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop_etiq = recortar(frame, x1, y1, x2, y2, MARGEN_ETIQ)
                if crop_etiq.size == 0:
                    continue

                t0 = time.time()
                nuevos, detalle_vin, zonas = analizar_pendientes(
                    crop_etiq, model_codigos, pendientes)
                t_cod = time.time() - t0

                zonas_activas[track_id] = zonas

                if nuevos or detalle_vin:
                    for k, v in nuevos.items():
                        estado[k] = v
                    log_actualizacion(frame_num, track_id, estado,
                                      nuevos, detalle_vin, t_cod)
                    if all(v is not None for v in estado.values()):
                        log_completo(track_id, estado, frame_num)
                        completados.add(track_id)

            # ── dibujar y grabar ──────────────────────────────────────────────
            vis = dibujar(frame, last_boxes, CONF_ETIQ, frame_num, fps_proc, tracks)
            if writer:
                writer.write(vis)
            last_vis = vis
            cv2.imshow("Debug codigos", vis)
            mostrar_crop_zonas(frame, last_boxes, CONF_ETIQ, tracks, zonas_activas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Saliendo.")
                break
            elif key == ord(' '):
                pausado = True
                print(f"PAUSADO en frame {frame_num}")
            elif key == ord('s'):
                p = os.path.join(OUTPUT_DIR, f"debug_f{frame_num:06d}.jpg")
                cv2.imwrite(p, vis)
                print(f"  [SCREENSHOT] {p}")

    except KeyboardInterrupt:
        print("\n[Interrumpido]")
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"\nVideo guardado en: {ruta_out}")
        cv2.destroyAllWindows()
        log_resumen(tracks, completados, frame_num)


if __name__ == "__main__":
    main()
