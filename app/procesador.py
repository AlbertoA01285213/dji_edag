#!/usr/bin/env python3

import os
import time
import sqlite3
import logging
import difflib
import threading
import numpy as np
from datetime import datetime
from collections import Counter
from fastapi import FastAPI, BackgroundTasks

app = FastAPI(title="Procesador de datos server")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProcesadorDatos")

conn = sqlite3.connect("datos.db")
cursor = conn.cursor()
cursor.execute("""
               CREATE TABLE IF NOT EXISTS etiquetas_procesadas (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
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

ESTADO_PROCESO = {
    "corriendo": False
}

DB_PATH = "datos.db"

def calcular_voto_mayoria_ocr(lista_textos):
    """Filtra textos vacíos, agrupa por similitud (>85%) y elige el string más estable"""
    limpios = [t for t in lista_textos if t and t != "—" and t.strip() != ""]
    if not limpios:
        return "—"
    
    conteo_crudo = Counter(limpios)
    votos_consolidados = {}
    
    # Agrupación por vecindarios de similitud (Algoritmo Levenshtein nativo)
    for texto, frecuencia in conteo_crudo.items():
        fusionado = False
        for master_text in votos_consolidados:
            # Comparamos qué tan similares son los dos textos (de 0.0 a 1.0)
            if difflib.SequenceMatcher(None, texto, master_text).ratio() > 0.85:
                votos_consolidados[master_text] += frecuencia
                fusionado = True
                break
        if not fusionado:
            votos_consolidados[texto] = frecuencia
            
    # Retornamos la cadena de texto que ganó el voto de la mayoría consolidada
    return max(votos_consolidados, key=votos_consolidados.get) if votos_consolidados else "—"


def bucle_procesamiento_datos():
    global ESTADO_PROCESO
    ESTADO_PROCESO["corriendo"] = True
    logger.info("Iniciando consolidación y deduplicación de registros...")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 1. Traer todos los datos crudos recolectados por YOLO
        cursor.execute("SELECT vin_barra, vin_ult, datamatrix_link, datamatrix_num, pkn_largo, cve_com, fecha_registro FROM etiquetas ORDER BY fecha_registro ASC")
        rows = cursor.fetchall()
        
        if not rows:
            logger.warning("No hay datos crudos en la tabla 'etiquetas' para procesar.")
            ESTADO_PROCESO["corriendo"] = False
            conn.close()
            return

        # 2. Agrupación por Ventana de Tiempo Activa (Máximo 5 segundos de separación entre capturas)
        grupos = []
        grupo_actual = []
        ultimo_timestamp = None

        for r in rows:
            # Convertir el string de la DB a un objeto datetime real de Python
            ts_actual = datetime.strptime(r[6], "%Y-%m-%d %H:%M:%S")
            
            if ultimo_timestamp is None:
                grupo_actual.append(r)
            else:
                diferencia_tiempo = (ts_actual - ultimo_timestamp).total_seconds()
                if diferencia_tiempo <= 5.0:
                    grupo_actual.append(r)
                else:
                    grupos.append(grupo_actual)
                    grupo_actual = [r]
            ultimo_timestamp = ts_actual
            
        if grupo_actual:
            grupos.append(grupo_actual)

        logger.info(f"Se detectaron {len(grupos)} ráfagas (etiquetas físicas individuales). Procesando votaciones...")

        # Coordenadas base simuladas (Trayectoria representativa en Monterrey)
        lat_base, lon_base = 25.584, -100.266

        # 3. Aplicar Voto de Mayoría a cada grupo y guardar el resultado limpio
        for i, grupo in enumerate(grupos):
            vins_barra = [item[0] for item in grupo]
            vins_ult = [item[1] for item in grupo]
            dm_links = [item[2] for item in grupo]
            dm_nums = [item[3] for item in grupo]
            pkns = [item[4] for item in grupo]
            cves = [item[5] for item in grupo]

            vin_ult_limpio = calcular_voto_mayoria_ocr(vins_ult)
            dm_link_limpio = calcular_voto_mayoria_ocr(dm_links)
            dm_num_limpio = calcular_voto_mayoria_ocr(dm_nums)
            pkn_limpio = calcular_voto_mayoria_ocr(pkns)
            cve_limpio = calcular_voto_mayoria_ocr(cves)

            # Generamos una pequeña dispersión geográfica para que los pines no se encimen en el mapa
            lat_final = lat_base + (i * 0.00015)
            lon_final = lon_base + (i * 0.00015)

            cursor.execute("""
                INSERT INTO etiquetas_procesadas (vin_ult, datamatrix_link, datamatrix_num, pkn_largo, cve_com, latitud, longitud)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (vin_ult_limpio, dm_link_limpio, dm_num_limpio, pkn_limpio, cve_limpio, lat_final, lon_final))

        conn.commit()
        conn.close()
        logger.info("Procesamiento y guardado finalizado con éxito.")
        
    except Exception as e:
        logger.error(f"Error durante el procesamiento: {e}")
        
    ESTADO_PROCESO["corriendo"] = False

PATH_BASE = os.path.join(os.path.expanduser('~'), 'Documents', 'dji_edag', 'app')

@app.post("/iniciar")
def iniciar_procesamiento(backgorund_tasks: BackgroundTasks):
    if ESTADO_PROCESO["corriendo"]:
        return{"status": "error", "message": "Ya hay un procesamiento en ejecucion."}
    
    backgorund_tasks.add_task(bucle_procesamiento_datos)
    return {"status": "succes", "message": "Procesamiento iniciado correctamente"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.3", port=8002)