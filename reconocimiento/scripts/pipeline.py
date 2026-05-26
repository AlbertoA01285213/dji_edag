import cv2
import os
import argparse
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO
from supabase import create_client
from leer_codigos import leer_codigos

load_dotenv()

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODELO_ETIQ    = os.path.join(BASE_DIR, "weights", "modelo_etiqueta.pt")
MODELO_CODIGOS = os.path.join(BASE_DIR, "weights", "modelo_codigos.pt")

CONF_ETIQUETA = 0.5
CONF_CODIGOS  = 0.4
MARGEN_ETIQ   = 10
MARGEN_ZONA   = 8
ZOOM_ANCHO    = 1280

GRACE_FRAMES      = 15
MAX_INTENTOS      = 30
INTERVALO_LECTURA = 5

CLASES = {
    0: "vin",
    1: "pkg_largo",
    2: "cve_com",
    3: "datamatrix_num",
    4: "datamatrix_link",
}


def recortar(imagen, x1, y1, x2, y2, margen=0):
    h, w = imagen.shape[:2]
    x1 = max(0, x1 - margen)
    y1 = max(0, y1 - margen)
    x2 = min(w, x2 + margen)
    y2 = min(h, y2 + margen)
    return imagen[y1:y2, x1:x2]


def nitidez(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def zoom_etiqueta(crop):
    h, w = crop.shape[:2]
    if w >= ZOOM_ANCHO:
        return crop
    scale = ZOOM_ANCHO / w
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def actualizar_mejor_crop(frame, box_etiq, estado):
    x1, y1, x2, y2 = map(int, box_etiq.xyxy[0])
    crop  = recortar(frame, x1, y1, x2, y2, MARGEN_ETIQ)
    score = nitidez(crop)
    if score > estado["mejor_nitidez"]:
        estado["mejor_nitidez"] = score
        estado["mejor_crop"]    = crop


def extraer_zonas(etiqueta, model_codigos):
    """Corre modelo_codigos y devuelve dict de crops por campo."""
    res    = model_codigos(etiqueta, conf=CONF_CODIGOS, verbose=False)[0]
    zonas  = {}
    for box in res.boxes:
        clase_id = int(box.cls[0])
        if clase_id not in CLASES:
            continue
        campo = CLASES[clase_id]
        if campo in zonas:
            continue
        cx1, cy1, cx2, cy2 = map(int, box.xyxy[0])
        zonas[campo] = recortar(etiqueta, cx1, cy1, cx2, cy2, MARGEN_ZONA)
    return zonas


def intentar_leer(model_codigos, estado):
    """Usa el mejor crop acumulado para leer los codigos."""
    if estado["mejor_crop"] is None:
        return

    etiqueta = zoom_etiqueta(estado["mejor_crop"])
    zonas    = extraer_zonas(etiqueta, model_codigos)

    resultado = leer_codigos(etiqueta, zonas=zonas if zonas else None)

    for campo, valor in resultado.items():
        if valor and estado["campos"].get(campo) is None:
            estado["campos"][campo] = valor


def insertar_registro(supabase, track_id, nombre_video, campos, frame_num):
    leidos      = sum(1 for v in campos.values() if v)
    nombre_base = os.path.splitext(nombre_video)[0]
    registro    = {
        'imagen':             f"{nombre_base}_track_{track_id}_f{frame_num}",
        'foto_original':      nombre_video,
        'confianza_etiqueta': None,
        'latitud':            None,
        'longitud':           None,
        **campos,
    }
    supabase.table('escaneos_etiquetas').insert(registro).execute()
    print(
        f"  [DB] track={track_id:>4} frame={frame_num:>6} | "
        f"{leidos}/5 | VIN:{campos['vin']} | PKN:{campos['pkg_largo']} | CVE:{campos['cve_com']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Pipeline etiquetas VW — Saishuu")
    parser.add_argument("video", help="Ruta al video (o pon el archivo en input/)")
    parser.add_argument("--skip",  type=int,   default=3,    help="1 de cada N frames (default: 3)")
    parser.add_argument("--conf",  type=float, default=0.5,  help="Confianza etiqueta (default: 0.5)")
    parser.add_argument("--imgsz", type=int,   default=1280, help="Tamano imagen (default: 1280)")
    args = parser.parse_args()

    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    if not url or not key:
        print("[ERROR] Configura SUPABASE_URL y SUPABASE_KEY en .env")
        return

    if not os.path.exists(args.video):
        print(f"[ERROR] Video no encontrado: {args.video}")
        return

    supabase      = create_client(url, key)
    model_etiq    = YOLO(MODELO_ETIQ)
    model_codigos = YOLO(MODELO_CODIGOS)

    cap          = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    nombre_video = os.path.basename(args.video)

    print(f"Video: {nombre_video} | {total_frames} frames | {fps:.1f} fps | skip={args.skip}")
    print("=" * 60)

    tracks    = {}
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if frame_num % args.skip != 0:
            continue

        res_etiq     = model_etiq.track(frame, conf=args.conf, imgsz=args.imgsz,
                                        persist=True, tracker="bytetrack.yaml",
                                        verbose=False)[0]
        ids_en_frame = set()

        if res_etiq.boxes.id is not None:
            for box_etiq in res_etiq.boxes:
                if box_etiq.id is None:
                    continue
                track_id = int(box_etiq.id[0])
                ids_en_frame.add(track_id)

                if track_id not in tracks:
                    tracks[track_id] = {
                        "campos":         {k: None for k in
                                           ("vin","pkg_largo","cve_com",
                                            "datamatrix_num","datamatrix_link")},
                        "intentos":       0,
                        "insertado":      False,
                        "ultimo_frame":   frame_num,
                        "ultimo_intento": 0,
                        "mejor_crop":     None,
                        "mejor_nitidez":  0.0,
                    }

                estado = tracks[track_id]
                estado["ultimo_frame"] = frame_num
                actualizar_mejor_crop(frame, box_etiq, estado)

                if estado["insertado"] or estado["intentos"] >= MAX_INTENTOS:
                    continue
                if frame_num - estado["ultimo_intento"] < INTERVALO_LECTURA:
                    continue

                intentar_leer(model_codigos, estado)
                estado["intentos"]      += 1
                estado["ultimo_intento"] = frame_num

                if all(v is not None for v in estado["campos"].values()):
                    insertar_registro(supabase, track_id, nombre_video,
                                      estado["campos"], frame_num)
                    estado["insertado"] = True

        for tid, estado in tracks.items():
            if (tid not in ids_en_frame
                    and not estado["insertado"]
                    and frame_num - estado["ultimo_frame"] > GRACE_FRAMES
                    and estado["campos"]["vin"] is not None):
                insertar_registro(supabase, tid, nombre_video,
                                  estado["campos"], estado["ultimo_frame"])
                estado["insertado"] = True

        if frame_num % (30 * args.skip) == 0:
            activos    = sum(1 for e in tracks.values() if not e["insertado"])
            insertados = sum(1 for e in tracks.values() if e["insertado"])
            pct        = frame_num / total_frames * 100 if total_frames else 0
            print(f"Frame {frame_num}/{total_frames} ({pct:.0f}%) | "
                  f"activos: {activos} | insertados: {insertados}")

    print("\nFlush final...")
    for tid, estado in tracks.items():
        if not estado["insertado"]:
            intentar_leer(model_codigos, estado)
            if estado["campos"]["vin"] is not None:
                insertar_registro(supabase, tid, nombre_video,
                                  estado["campos"], estado["ultimo_frame"])
                estado["insertado"] = True

    cap.release()

    insertados = sum(1 for e in tracks.values() if e["insertado"])
    sin_vin    = sum(1 for e in tracks.values()
                     if not e["insertado"] and e["campos"]["vin"] is None)
    print(f"\nPipeline completo.")
    print(f"  Vehiculos insertados: {insertados}")
    print(f"  Detectados sin VIN (descartados): {sin_vin}")
    print(f"  Total tracks: {len(tracks)}")


if __name__ == '__main__':
    main()
