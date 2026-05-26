"""
debug_etiqueta.py — Visualiza deteccion de etiquetas en video sin tocar DB ni codigos.

Controles:
  ESPACIO  pausar / reanudar
  q        salir
  s        guardar frame actual en output/debug_fXXXXX.jpg
  + / -    subir / bajar confianza 0.05 en vivo
"""

import cv2
import os
import sys
import numpy as np
from ultralytics import YOLO

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELO_ETIQ = os.path.join(BASE_DIR, "weights", "modelo_etiqueta.pt")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO = os.path.join(BASE_DIR, "input", "DJI_20260519132308_0122_D.MP4")
if len(sys.argv) > 1:
    VIDEO = sys.argv[1]

SKIP       = 2      # procesa 1 de cada N frames (mas rapido)
CONF_INIT  = 0.4
IMGSZ      = 1280
SHOW_CROP  = True   # muestra recorte de la etiqueta en ventana aparte


def nitidez(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def dibujar(frame, boxes, conf_umbral, frame_num, fps_proc):
    vis = frame.copy()
    h, w = vis.shape[:2]

    detecciones = 0
    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_umbral:
            continue
        detecciones += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if box.id is not None else -1

        # Color segun confianza
        color = (0, int(255 * conf), int(255 * (1 - conf)))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        crop  = frame[max(0, y1):y2, max(0, x1):x2]
        nit   = nitidez(crop) if crop.size > 0 else 0
        label = f"ID:{track_id} {conf:.2f} nit:{nit:.0f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        by = max(y1 - 4, th + 4)
        cv2.rectangle(vis, (x1, by - th - 4), (x1 + tw + 4, by + 2), color, -1)
        cv2.putText(vis, label, (x1 + 2, by - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # HUD
    hud = (f"Frame {frame_num} | det={detecciones} | "
           f"conf>={conf_umbral:.2f} | {fps_proc:.1f} fps proc")
    cv2.putText(vis, hud, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return vis, detecciones


def mostrar_crop(frame, boxes, conf_umbral):
    """Muestra el mejor crop de etiqueta en ventana separada."""
    mejor = None
    mejor_nit = -1
    for box in boxes:
        if float(box.conf[0]) < conf_umbral:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        nit = nitidez(crop)
        if nit > mejor_nit:
            mejor_nit = nit
            mejor = crop

    if mejor is not None:
        # Escalar para que sea visible sin ser enorme
        mh, mw = mejor.shape[:2]
        target_w = 640
        if mw > 0:
            scale = target_w / mw
            preview = cv2.resize(mejor, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_LINEAR)
            cv2.putText(preview, f"nit={mejor_nit:.0f}", (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow("Crop etiqueta", preview)
    else:
        # Frame negro cuando no hay deteccion
        blank = np.zeros((200, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Sin deteccion", (180, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
        cv2.imshow("Crop etiqueta", blank)


def main():
    if not os.path.exists(VIDEO):
        print(f"[ERROR] Video no encontrado: {VIDEO}")
        print("Uso: python debug_etiqueta.py [ruta_video]")
        return

    print(f"Cargando modelo: {MODELO_ETIQ}")
    model = YOLO(MODELO_ETIQ)

    cap         = cv2.VideoCapture(VIDEO)
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video   = cap.get(cv2.CAP_PROP_FPS)
    nombre      = os.path.basename(VIDEO)
    conf        = CONF_INIT
    pausado     = False
    frame_num   = 0
    fps_proc    = 0.0
    last_boxes  = []

    print(f"Video : {nombre}")
    print(f"Frames: {total} | FPS: {fps_video:.1f} | skip={SKIP} | conf={conf}")
    print("Controles: ESPACIO=pausa  q=salir  s=guardar  +/- confianza")
    print("=" * 60)

    import time
    t0 = time.time()

    while True:
        if not pausado:
            ret, frame = cap.read()
            if not ret:
                print("\nFin del video.")
                break

            frame_num += 1

            if frame_num % SKIP != 0:
                # mostrar frame sin procesar para no saltarse visualmente
                vis, _ = dibujar(frame, last_boxes, conf, frame_num, fps_proc)
                if SHOW_CROP:
                    mostrar_crop(frame, last_boxes, conf)
                cv2.imshow("Debug etiqueta", vis)
            else:
                t_ini = time.time()
                res = model.track(frame, conf=conf, imgsz=IMGSZ,
                                  persist=True, tracker="bytetrack.yaml",
                                  verbose=False)[0]
                dt      = time.time() - t_ini
                fps_proc = 1.0 / dt if dt > 0 else 0.0

                last_boxes = res.boxes if res.boxes.id is not None else []
                n_det      = len(last_boxes)

                if n_det > 0:
                    for box in last_boxes:
                        c  = float(box.conf[0])
                        tid = int(box.id[0]) if box.id is not None else -1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cr  = frame[max(0,y1):y2, max(0,x1):x2]
                        nit = nitidez(cr) if cr.size > 0 else 0
                        print(f"  f{frame_num:>6} | ID:{tid:>3} | "
                              f"conf={c:.2f} | nit={nit:>7.1f} | "
                              f"bbox=[{x1},{y1},{x2},{y2}]")

                vis, _ = dibujar(frame, last_boxes, conf, frame_num, fps_proc)
                if SHOW_CROP:
                    mostrar_crop(frame, last_boxes, conf)
                cv2.imshow("Debug etiqueta", vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Saliendo.")
            break
        elif key == ord(' '):
            pausado = not pausado
            print(f"{'PAUSADO' if pausado else 'REANUDADO'} en frame {frame_num}")
        elif key == ord('s'):
            out_path = os.path.join(OUTPUT_DIR, f"debug_f{frame_num:06d}.jpg")
            cv2.imwrite(out_path, vis)
            print(f"  [GUARDADO] {out_path}")
        elif key == ord('+') or key == ord('='):
            conf = min(0.95, round(conf + 0.05, 2))
            print(f"  Confianza -> {conf:.2f}")
        elif key == ord('-'):
            conf = max(0.05, round(conf - 0.05, 2))
            print(f"  Confianza -> {conf:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    total_time = time.time() - t0
    print(f"\nTiempo total: {total_time:.1f}s | Frames procesados: {frame_num}")


if __name__ == "__main__":
    main()
