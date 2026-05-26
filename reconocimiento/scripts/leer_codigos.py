import cv2
import re
import numpy as np
import zxingcpp

PATRON_VIN     = re.compile(r'^[A-HJ-NPR-Z0-9]{17}$')
PATRON_PKN     = re.compile(r'^\d{13,15}$')
PATRON_CVE_COM = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{2}$')
PATRON_DM_OK   = re.compile(r'^[\x20-\x7E]{4,}$')

KERNEL_SHARP = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]], dtype=np.float32)

PADDING_1D = 40   # zona silenciosa requerida por Code39

ROTACIONES = [None,
              cv2.ROTATE_90_CLOCKWISE,
              cv2.ROTATE_180,
              cv2.ROTATE_90_COUNTERCLOCKWISE]

TARGET_1D = 2400
TARGET_DM = 600

# Zonas hardcoded como fallback (x1%, y1%, x2%, y2%)
ZONAS_1D_FB = {
    'vin': (0.02, 0.08, 0.92, 0.28),
    'pkn': (0.01, 0.60, 0.85, 0.85),
    'cve': (0.01, 0.70, 0.52, 0.92),
}
ZONAS_DM_FB = {
    'dm1': (0.68, 0.18, 0.99, 0.48),
    'dm2': (0.01, 0.30, 0.25, 0.60),
}


def _ordenar_puntos(pts):
    """Ordena 4 puntos como [top-left, top-right, bottom-right, bottom-left]."""
    pts  = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s        = pts.sum(axis=1)
    rect[0]  = pts[np.argmin(s)]   # top-left:     suma minima
    rect[2]  = pts[np.argmax(s)]   # bottom-right: suma maxima
    d        = np.diff(pts, axis=1)
    rect[1]  = pts[np.argmin(d)]   # top-right:    diff minima
    rect[3]  = pts[np.argmax(d)]   # bottom-left:  diff maxima
    return rect


def rectificar_etiqueta(crop_bgr):
    """
    Detecta el contorno de la etiqueta por contraste, corrige la perspectiva
    y devuelve (imagen_rectificada, exito).

    Si no puede detectar un cuadrilatero claro, devuelve (crop_bgr, False).
    """
    h, w = crop_bgr.shape[:2]

    # Filtro bilateral: suaviza ruido interno pero preserva el borde fuerte
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    # Canny para encontrar bordes de la etiqueta
    edges  = cv2.Canny(blur, 25, 90)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges  = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop_bgr, False

    # Solo considerar contornos que cubran al menos 15% del area total
    area_min   = h * w * 0.15
    candidatos = [c for c in contours if cv2.contourArea(c) > area_min]
    if not candidatos:
        return crop_bgr, False

    mayor  = max(candidatos, key=cv2.contourArea)
    peri   = cv2.arcLength(mayor, True)
    approx = cv2.approxPolyDP(mayor, 0.02 * peri, True)

    if len(approx) == 4:
        pts = _ordenar_puntos(approx)
    else:
        # Fallback: minAreaRect siempre da 4 esquinas
        rect = cv2.minAreaRect(mayor)
        box  = cv2.boxPoints(rect)
        pts  = _ordenar_puntos(box)

    tl, tr, br, bl = pts
    dst_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    dst_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    if dst_w < 50 or dst_h < 50:
        return crop_bgr, False

    # Orientacion vertical: la etiqueta VW es mas ancha que alta desde el dron,
    # la rotamos para que quede alta > ancha (portrait)
    if dst_w > dst_h:
        dst_w, dst_h = dst_h, dst_w
        pts = np.array([tr, br, bl, tl], dtype=np.float32)  # giro 90° CCW

    dst = np.array([
        [0,         0        ],
        [dst_w - 1, 0        ],
        [dst_w - 1, dst_h - 1],
        [0,         dst_h - 1],
    ], dtype=np.float32)

    M      = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(crop_bgr, M, (dst_w, dst_h))
    return warped, True


def _escalar(img, target_w):
    if img.shape[1] < target_w:
        f = target_w / img.shape[1]
        return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return img


def _variantes(gris):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gc    = clahe.apply(gris)
    ns    = cv2.filter2D(gc, -1, KERNEL_SHARP)
    _, otsu = cv2.threshold(gc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [
        gris,
        gc,
        ns,
        otsu,
        cv2.adaptiveThreshold(gc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 21, 6),
    ]


def _con_padding(img, pad=PADDING_1D):
    """Agrega margen blanco — zona silenciosa requerida por Code39."""
    return cv2.copyMakeBorder(img, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=255)


def _rotar(img, rot):
    return img if rot is None else cv2.rotate(img, rot)


def _crop_zona(img, zona):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = zona
    r = img[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
    return r if r.size > 0 else None


_BINARIZERS_1D = [
    zxingcpp.Binarizer.LocalAverage,
    zxingcpp.Binarizer.GlobalHistogram,
    zxingcpp.Binarizer.FixedThreshold,
]


def _decode_1d(gris):
    encontrados = set()
    for v in _variantes(gris):
        vp = _con_padding(v)
        for binarizer in _BINARIZERS_1D:
            for c in zxingcpp.read_barcodes(
                    vp, try_rotate=True, try_invert=True, binarizer=binarizer):
                txt = c.text.strip()
                if txt:
                    encontrados.add(txt)
            for c in zxingcpp.read_barcodes(
                    vp, formats=zxingcpp.BarcodeFormat.Code39,
                    try_rotate=True, try_invert=True, binarizer=binarizer):
                txt = c.text.strip()
                if txt:
                    encontrados.add(txt)
    return encontrados


def _decode_dm(gris):
    encontrados = []
    for v in _variantes(gris):
        for c in zxingcpp.read_barcodes(v, formats=zxingcpp.BarcodeFormat.DataMatrix,
                                        try_rotate=True):
            txt = c.text.strip()
            if txt and PATRON_DM_OK.match(txt) and txt not in encontrados:
                encontrados.append(txt)
    return encontrados


def _escanear_imagen_completa(img_bgr):
    todos_1d = set()
    todos_dm = []

    for rot in ROTACIONES:
        g = cv2.cvtColor(_rotar(img_bgr, rot), cv2.COLOR_BGR2GRAY)
        g = _escalar(g, TARGET_1D)
        todos_1d |= _decode_1d(g)

        g_dm = _escalar(cv2.cvtColor(_rotar(img_bgr, rot), cv2.COLOR_BGR2GRAY), TARGET_DM)
        for dm in _decode_dm(g_dm):
            if dm not in todos_dm:
                todos_dm.append(dm)

        vin_ok = any(PATRON_VIN.match(t)     for t in todos_1d)
        pkn_ok = any(PATRON_PKN.match(t)     for t in todos_1d)
        cve_ok = any(PATRON_CVE_COM.match(t) for t in todos_1d)
        if vin_ok and pkn_ok and cve_ok and len(todos_dm) >= 2:
            break

    return todos_1d, todos_dm


def _escanear_zona_1d(img_bgr, zona_bgr=None, zona_pct=None):
    """Escanea una zona 1D. Acepta crop directo o porcentaje sobre img_bgr."""
    for rot in ROTACIONES:
        if zona_bgr is not None:
            c = _rotar(zona_bgr, rot)
        else:
            c = _crop_zona(_rotar(img_bgr, rot), zona_pct)
        if c is None:
            continue
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c
        g = _escalar(g, TARGET_1D)
        found = _decode_1d(g)
        if found:
            return found
    return set()


def _escanear_zona_dm(img_bgr, zona_bgr=None, zona_pct=None):
    """Escanea una zona DataMatrix. Acepta crop directo o porcentaje."""
    for rot in ROTACIONES:
        if zona_bgr is not None:
            c = _rotar(zona_bgr, rot)
        else:
            c = _crop_zona(_rotar(img_bgr, rot), zona_pct)
        if c is None:
            continue
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c
        g = _escalar(g, TARGET_DM)
        found = _decode_dm(g)
        if found:
            return found[0]
    return None


def leer_codigos(label_bgr, zonas=None):
    """
    Lee los 5 codigos de una etiqueta VW.

    label_bgr : recorte BGR de la etiqueta completa
    zonas     : dict opcional con crops BGR pre-detectados por modelo_codigos
                {'vin': crop, 'pkg_largo': crop, 'cve_com': crop,
                 'datamatrix_num': crop, 'datamatrix_link': crop}
    """
    # Rectificar perspectiva antes de escanear
    rectificada, ok = rectificar_etiqueta(label_bgr)
    label_bgr = rectificada if ok else label_bgr

    # Paso 1: escaneo completo de la imagen en 4 rotaciones
    todos_1d, todos_dm = _escanear_imagen_completa(label_bgr)

    vin = pkn = cve = dm1 = dm2 = None
    for txt in todos_1d:
        if not vin and PATRON_VIN.match(txt):     vin = txt
        if not pkn and PATRON_PKN.match(txt):     pkn = txt
        if not cve and PATRON_CVE_COM.match(txt): cve = txt
    if len(todos_dm) >= 1: dm1 = todos_dm[0]
    if len(todos_dm) >= 2: dm2 = todos_dm[1]

    # Paso 2: zonas especificas para lo que fallo
    # Usa crops de modelo_codigos si estan disponibles, si no usa porcentajes
    if not vin:
        crop_vin = zonas.get('vin') if zonas else None
        found = _escanear_zona_1d(label_bgr, zona_bgr=crop_vin,
                                  zona_pct=ZONAS_1D_FB['vin'] if crop_vin is None else None)
        vin = next((t for t in found if PATRON_VIN.match(t)), None)

    if not pkn:
        crop_pkn = zonas.get('pkg_largo') if zonas else None
        found = _escanear_zona_1d(label_bgr, zona_bgr=crop_pkn,
                                  zona_pct=ZONAS_1D_FB['pkn'] if crop_pkn is None else None)
        pkn = next((t for t in found if PATRON_PKN.match(t)), None)

    if not cve:
        crop_cve = zonas.get('cve_com') if zonas else None
        found = _escanear_zona_1d(label_bgr, zona_bgr=crop_cve,
                                  zona_pct=ZONAS_1D_FB['cve'] if crop_cve is None else None)
        cve = next((t for t in found if PATRON_CVE_COM.match(t)), None)

    if not dm1:
        crop_dm1 = zonas.get('datamatrix_num') if zonas else None
        dm1 = _escanear_zona_dm(label_bgr, zona_bgr=crop_dm1,
                                zona_pct=ZONAS_DM_FB['dm1'] if crop_dm1 is None else None)

    if not dm2:
        crop_dm2 = zonas.get('datamatrix_link') if zonas else None
        dm2 = _escanear_zona_dm(label_bgr, zona_bgr=crop_dm2,
                                zona_pct=ZONAS_DM_FB['dm2'] if crop_dm2 is None else None)

    return {
        'vin':             vin,
        'pkg_largo':       pkn,
        'cve_com':         cve,
        'datamatrix_num':  dm1,
        'datamatrix_link': dm2,
    }
