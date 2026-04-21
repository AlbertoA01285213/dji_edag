import os
import cv2
import numpy as np
import sqlite3
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import cv2

# --- CONFIGURACIÓN ---
PATH_DJI = Path.home() / "Downloads" / "dji"
DB_NAME = "imagenes_dji.db"
MARGEN = 0.3

def get_decimal_from_dms(dms, ref):
    """Convierte coordenadas EXIF a grados decimales."""
    degrees = dms[0]
    minutes = dms[1]
    seconds = dms[2]
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal


def get_exif_data(image_path):
    """Extrae las coordenadas GPS de la imagen."""
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if not exif_data:
                return None, None

            gps_info = {}
            for tag, value in exif_data.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    for t in value:
                        sub_tag = GPSTAGS.get(t, t)
                        gps_info[sub_tag] = value[t]

            if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
                lat = get_decimal_from_dms(gps_info["GPSLatitude"], gps_info["GPSLatitudeRef"])
                lon = get_decimal_from_dms(gps_info["GPSLongitude"], gps_info["GPSLongitudeRef"])
                return lat, lon
    except Exception as e:
        print(f"  ⚠️ Error leyendo EXIF: {e}")
    
    return None, None


def read_qr(image_path):
    """Lee el código QR de la imagen usando OpenCV."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return "Error: No se pudo leer la imagen"
        
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(img)
        
        # Verificar si se encontró un QR
        if not data:
            return "No QR found"
        
        # Verificar si points es válido y tiene la estructura correcta
        if points is not None and len(points) > 0:
            # Calcular el centro del QR
            qr_center_x = np.mean(points[0][:, 0])
            h, w = img.shape[:2]
            
            limite_izquierdo = MARGEN
            limite_derecho = 1 - MARGEN
            
            # Verificar si está dentro de los márgenes
            if qr_center_x < w * limite_izquierdo or qr_center_x > w * limite_derecho:
                return "No QR found"
        
        return data
    
    except Exception as e:
        print(f"  ⚠️ Error leyendo QR: {e}")
        return "Error reading QR"


def init_db():
    """Inicializa la base de datos SQLite."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fotos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre_archivo TEXT,
            qr_contenido TEXT,
            latitud REAL,
            longitud REAL
        )
    ''')
    conn.commit()
    return conn

def main():
    if not PATH_DJI.exists():
        print(f"❌ La carpeta {PATH_DJI} no existe.")
        return

    conn = init_db()
    cursor = conn.cursor()
    
    print(f"🚀 Procesando imágenes en: {PATH_DJI}...")
    print("-" * 50)

    # Extensiones comunes de DJI
    extensions = ("*.jpg", "*.jpeg", "*.JPG", "*.PNG", "*.png")
    files = []
    for ext in extensions:
        files.extend(PATH_DJI.glob(ext))

    if not files:
        print("⚠️ No se encontraron imágenes en la carpeta.")
        return

    for img_path in files:
        try:
            print(f"📸 Analizando {img_path.name}...")
            
            # 1. Obtener QR
            qr_data = read_qr(img_path)
            
            # 2. Obtener Coordenadas
            lat, lon = get_exif_data(img_path)
            
            # 3. Guardar en DB
            cursor.execute('''
                INSERT INTO fotos (nombre_archivo, qr_contenido, latitud, longitud)
                VALUES (?, ?, ?, ?)
            ''', (img_path.name, qr_data, lat, lon))
            
            conn.commit()
        except Exception as e:
            print(f"⚠️ Error procesando {img_path.name}: {e}")

        print("-" * 30)

    conn.close()
    print("\n✅ Proceso terminado. Datos guardados en", DB_NAME)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM fotos")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM fotos WHERE qr_contenido != 'No QR found' AND qr_contenido NOT LIKE 'Error%'")
    with_qr = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM fotos WHERE latitud IS NOT NULL")
    with_gps = cursor.fetchone()[0]
    conn.close()
    
    print(f"\n📊 Resumen:")
    print(f"   Total imágenes procesadas: {total}")
    print(f"   Con QR válido: {with_qr}")
    print(f"   Con coordenadas GPS: {with_gps}")

if __name__ == "__main__":
    main()
