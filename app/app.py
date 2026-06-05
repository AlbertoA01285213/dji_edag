#!/usr/bin/env python3
import os
import sys
import cv2
import json
import base64
import sqlite3
import requests
import folium
import numpy as np

from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QCursor
from PySide6.QtCore import Signal, QThread, Qt, Slot, QPoint, QObject, QTimer
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QToolTip, QComboBox, 
                             QVBoxLayout, QHBoxLayout, QWidget, QStackedWidget, QMessageBox,
                             QLabel, QFrame, QGridLayout, QSpinBox, QDoubleSpinBox, QWidget)
from PySide6.QtWebEngineWidgets import QWebEngineView


class AnalisisSignals(QObject):
    """Clase puente para emitir datos desde el hilo de YOLO hacia la GUI"""
    nuevo_frame_principal = Signal(QImage)
    nuevo_codigo_detectado = Signal(str, str, QImage)

class ZonaArrastrarVideo(QFrame):
    def __init__(self, al_seleccionar_archivo_callback):
        super().__init__()
        self.callback_archivo = al_seleccionar_archivo_callback
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Sunken)
        self.setStyleSheet("""
            QFrame { border: 2px dashed #3498db; border-radius: 8px; background-color: #ecf0f1; }
            QFrame:hover { background-color: #e8f4f8; border: 2px dashed #2980b9; }
        """)
        self.setMinimumHeight(120)
        self.setAcceptDrops(True)

        layout = QVBoxLayout()
        self.label_info = QLabel("Arrastra y suelta tu archivo de VIDEO aquí\n(.mp4, .avi, .mkv, .mov)")
        self.label_info.setAlignment(Qt.AlignCenter)
        self.label_info.setStyleSheet("color: #7f8c8d; font-size: 13px; font-weight: bold; border: none; background: transparent;")
        layout.addWidget(self.label_info)
        self.setLayout(layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile():
                event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            ruta_archivo = url.toLocalFile()
            if ruta_archivo.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                self.callback_archivo(ruta_archivo)
                event.acceptProposedAction()
            else:
                # CORRECCIÓN: Estilo explícito para evitar texto blanco oculto
                msg = QMessageBox(QMessageBox.Warning, "Archivo no válido", "Por favor, arrastra solo archivos de video (.mp4, .avi, .mkv, .mov).", parent=self)
                msg.setStyleSheet("QLabel{ color: #2c3e50; } QPushButton{ background-color: #dcdde1; color: black; }")
                msg.exec()


class ZonaArrastrarSRT(QFrame):
    def __init__(self, al_seleccionar_archivo_callback):
        super().__init__()
        self.callback_archivo = al_seleccionar_archivo_callback
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Sunken)
        self.setStyleSheet("""
            QFrame { border: 2px dashed #e67e22; border-radius: 8px; background-color: #ecf0f1; }
            QFrame:hover { background-color: #fdf2e9; border: 2px dashed #d35400; }
        """)
        self.setMinimumHeight(120)
        self.setAcceptDrops(True)

        layout = QVBoxLayout()
        self.label_info = QLabel("Arrastra y suelta tu archivo de TELEMETRÍA aquí\n(.srt)")
        self.label_info.setAlignment(Qt.AlignCenter)
        self.label_info.setStyleSheet("color: #7f8c8d; font-size: 13px; font-weight: bold; border: none; background: transparent;")
        layout.addWidget(self.label_info)
        self.setLayout(layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile():
                event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            ruta_archivo = url.toLocalFile()
            # CORRECCIÓN: Validación estricta para archivos de subtítulos
            if ruta_archivo.lower().endswith('.srt'):
                self.callback_archivo(ruta_archivo)
                event.acceptProposedAction()
            else:
                # CORRECCIÓN: Estilo explícito para visibilidad
                msg = QMessageBox(QMessageBox.Warning, "Archivo no válido", "Por favor, arrastra un archivo de telemetría válido con extensión .srt", parent=self)
                msg.setStyleSheet("QLabel{ color: #2c3e50; } QPushButton{ background-color: #dcdde1; color: black; }")
                msg.exec()

class MapaDashboard(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Sunken)
        self.setMinimumHeight(500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.vista_web = QWebEngineView()
        layout.addWidget(self.vista_web)

        self.cargar_mapa()

    def cargar_mapa(self):
        data = self.obtener_data_db()

        if not data:
            lat_inicial, lon_inicial, = 25.584, -100.266
        else:
            lat_inicial = data[0]['lat']
            lon_inicial = data[0]['lon']

        mapa = folium.Map(location=[lat_inicial, lon_inicial], zoom_start=17, tiles="CartoDB dark_matter")

        
        for et in data:
            # html_info = f"""
            # <div style="font-family: Arial; color: #white; background-color: #2c3e50; padding: 10px; border-radius: 5px;">
            #     <b>VIN_ULT:</b> {et['vin_ult']}<br>
            #     <b>DM_LINK:</b> {et['datamatrix_link']}<br>
            #     <b>DM_NUM:</b> {et['datamatrix_num']}<br>
            #     <b>PKN_LA:</b> {et['pkn_largo']}<br>
            #     <b>CVE_COM:</b> {et['cve_com']}<br>
            # </div>
            # """

            html_info = f"""
            <div style="font-family: 'Segoe UI', Arial; color: white; background-color: #2c3e50; padding: 12px; border-radius: 6px; min-width: 200px; box-shadow: 2px 2px 10px rgba(0,0,0,0.5);">
                <span style="color: #1abc9c; font-weight: bold; font-size: 14px;">Etiqueta Detectada</span><hr style="border: 0; border-top: 1px solid #7f8c8d; margin: 6px 0;">
                <b>VIN (Últ):</b> {et['vin_ult']}<br>
                <b>DM Link:</b> <span style="font-size: 11px; color: #3498db;">{et['datamatrix_link']}</span><br>
                <b>DM Núm:</b> {et['datamatrix_num']}<br>
                <b>PKN Largo:</b> {et['pkn_largo']}<br>
                <b>CVE COM:</b> <span style="background-color: #e67e22; padding: 2px 5px; border-radius: 3px; font-size: 11px;">{et['cve_com']}</span>
            </div>
            """

            folium.CircleMarker(
                location=[et["lat"], et["lon"]],
                radius=8,
                color="#1abc9c",
                fill=True,
                fill_color="#1abc9c",
                tooltip=folium.Tooltip(html_info, sticky=True)
            ).add_to(mapa)

        html_puro = mapa._repr_html_()
        self.vista_web.setHtml(html_puro)


    def obtener_data_db(self):
        detecciones = []
        if not os.path.exists("datos.db"):
            return detecciones

        try:
            conn = sqlite3.connect("datos.db")
            cursor = conn.cursor()
            # Validamos que la tabla exista antes de intentar consultar para prevenir fallos en la UI
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='etiquetas_procesadas'")
            if not cursor.fetchone():
                conn.close()
                return detecciones

            cursor.execute("SELECT id, vin_ult, datamatrix_link, datamatrix_num, pkn_largo, cve_com, latitud, longitud FROM etiquetas_procesadas")
            rows = cursor.fetchall()
            for r in rows:
                detecciones.append({
                    "id": r[0],
                    "vin_ult": r[1],
                    "datamatrix_link": r[2], # Corregido: Se añade la 'r' faltante
                    "datamatrix_num": r[3],
                    "pkn_largo": r[4],
                    "cve_com": r[5],
                    "lat": float(r[6]) if r[6] else 0.0,
                    "lon": float(r[7]) if r[7] else 0.0
                })
            conn.close()
        except Exception as e:
            print(f"Error crítico leyendo DB desde el Dashboard: {e}")
        return detecciones


class DroneDashboard(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Drone Mission Control v1.0")
        self.resize(1000, 650)

        self.video_seleccionado = None
        self.srt_seleccionado = None

        self.api_url_analizador = "http://127.0.0.1:8000"
        self.api_url_graficador = "http://127.0.0.2:8001"
        self.api_url_procesador = "http://127.0.0.3:8002"

        self.timer_actualizador = QTimer()
        self.timer_actualizador.setInterval(60) # Actualiza a ~16 FPS la GUI
        self.timer_actualizador.timeout.connect(self.solicitar_actualizacion_servidor_analisis) 
        self.timer_actualizador.timeout.connect(self.solicitar_actualizacion_servidor_grafica)       

        # Layout Principal
        self.main_layout = QHBoxLayout()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(self.main_layout)

        # Barra lateral
        self.sidebar = QVBoxLayout()
        self.sidebar.addWidget(QLabel("<b>Seleccionar menu</b>"))
        self.sidebar.addSpacing(10)
        self.btn_analisis = QPushButton("⚙️ Analisis")
        self.btn_resultados = QPushButton("📊 Resultados")

        self.sidebar.addWidget(self.btn_analisis)
        self.sidebar.addWidget(self.btn_resultados)

        self.sidebar.addStretch()
        self.main_layout.addLayout(self.sidebar, 1)

        self.pages = QStackedWidget()
        self.main_layout.addWidget(self.pages, 4)

        self.init_pages()

        self.btn_analisis.clicked.connect(lambda: self.pages.setCurrentIndex(0))
        # self.btn_resultados.clicked.connect(lambda: self.pages.setCurrentIndex(2))
        self.btn_resultados.clicked.connect(self.cambiar_a_pagina_resultados)

    
    def init_pages(self):
        # =========================================================================
        # PAGINA 1: Configuración de análisis
        # =========================================================================
        self.pagina_configuracion = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("<h2>Seleccion de datos</h2>"))

        layout.addWidget(QLabel("Selecciona el video"))
        layout.addSpacing(15)
        self.zona_drag_drop_video = ZonaArrastrarVideo(self.actualizar_video_seleccionado)
        layout.addWidget(self.zona_drag_drop_video)
        layout.addSpacing(15)

        self.label_video = QLabel("<b>Video seleccionado:</b> Ninguno (Usa el buscador o arrastra un archivo)")
        self.label_video.setStyleSheet("color: #c0392b; font-size: 13px; padding: 5px; background-color: #fafdff; border: 1px solid #dcdde1;")
        layout.addWidget(self.label_video)

        layout.addSpacing(20)

        self.zona_drag_drop_srt = ZonaArrastrarSRT(self.actualizar_srt_seleccionado)
        layout.addWidget(self.zona_drag_drop_srt)
        layout.addSpacing(15)

        self.label_srt = QLabel("<b>SRT seleccionado:</b> Ninguno (Usa el buscador o arrastra un archivo)")
        self.label_srt.setStyleSheet("color: #c0392b; font-size: 13px; padding: 5px; background-color: #fafdff; border: 1px solid #dcdde1;")
        layout.addWidget(self.label_srt)

        layout.addSpacing(30)

        self.btn_iniciar_analisis = QPushButton("Iniciar analisis")
        self.btn_iniciar_analisis.setStyleSheet("background-color: #2ecc71; font-weight: bold; font-size: 14px; height: 45px; color: white;")
        self.btn_iniciar_analisis.clicked.connect(self.iniciar_analisis)
        layout.addWidget(self.btn_iniciar_analisis)

        layout.addStretch()
        self.pagina_configuracion.setLayout(layout)
        self.pages.addWidget(self.pagina_configuracion)


        # =========================================================================
        # PAGINA 2: Vista de Análisis (La página "secreta" o activa de reproducción)
        # =========================================================================
        self.pagina_analisis = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("<h2>Analisis</h2>"))

        grid_frames = QGridLayout()

        # Columna izquierda
        self.main_frame = QLabel("Esperando video")
        self.main_frame.setMaximumSize(420, 230) # 640 360
        # self.main_frame.setStyleSheet("border: 2px solid gray;")
        self.main_frame.setStyleSheet("border: 2px solid #7f8c8d; background-color: #2c3e50; color: white; font-weight: bold;")
        self.main_frame.setAlignment(Qt.AlignCenter)
        grid_frames.addWidget(self.main_frame, 0, 0, 2, 1)

        self.graph_frame = QLabel("Grafica posicion")
        self.graph_frame.setMaximumSize(420, 230)
        # self.graph_frame.setStyleSheet("border: 2px solid gray;")
        self.graph_frame.setStyleSheet("border: 2px solid #7f8c8d; background-color: #2c3e50; color: white; font-weight: bold;")
        self.graph_frame.setAlignment(Qt.AlignCenter)
        grid_frames.addWidget(self.graph_frame, 2, 0, 2, 1)

        # Columna derecha
        self.vin_ult_frame = QLabel("Codigo de vin_ult")
        self.vin_ult_frame.setMaximumSize(250, 150)
        # self.vin_ult_frame.setStyleSheet("border: 2px solid gray;")
        self.vin_ult_frame.setStyleSheet("border: 2px solid #7f8c8d; background-color: #2c3e50; color: white; font-weight: bold;")
        self.vin_ult_frame.setAlignment(Qt.AlignCenter)
        grid_frames.addWidget(self.vin_ult_frame, 0, 1)

        self.datamatrix_link_frame = QLabel("Codigo datamatrix link")
        self.datamatrix_link_frame.setMaximumSize(250, 150)
        # self.datamatrix_link_frame.setStyleSheet("border: 2px solid gray;")
        self.datamatrix_link_frame.setStyleSheet("border: 2px solid #7f8c8d; background-color: #2c3e50; color: white; font-weight: bold;")
        self.datamatrix_link_frame.setAlignment(Qt.AlignCenter)
        grid_frames.addWidget(self.datamatrix_link_frame, 1, 1)

        self.datamatrix_num_frame = QLabel("Codigo datamatrix num")
        self.datamatrix_num_frame.setMaximumSize(250, 150)
        # self.datamatrix_num_frame.setStyleSheet("border: 2px solid gray;")
        self.datamatrix_num_frame.setStyleSheet("border: 2px solid #7f8c8d; background-color: #2c3e50; color: white; font-weight: bold;")
        self.datamatrix_num_frame.setAlignment(Qt.AlignCenter)
        grid_frames.addWidget(self.datamatrix_num_frame, 2, 1)

        self.pkn_largo_frame = QLabel("Codigo pkn largo")
        self.pkn_largo_frame.setMaximumSize(250, 150)
        # self.pkn_largo_frame.setStyleSheet("border: 2px solid gray;")
        self.pkn_largo_frame.setStyleSheet("border: 2px solid #7f8c8d; background-color: #2c3e50; color: white; font-weight: bold;")
        self.pkn_largo_frame.setAlignment(Qt.AlignCenter)
        grid_frames.addWidget(self.pkn_largo_frame, 0, 2)

        self.cve_com_frame = QLabel("Codigo cve com")
        self.cve_com_frame.setMaximumSize(250, 150)
        # self.cve_com_frame.setStyleSheet("border: 2px solid gray;")
        self.cve_com_frame.setStyleSheet("border: 2px solid #7f8c8d; background-color: #2c3e50; color: white; font-weight: bold;")
        self.cve_com_frame.setAlignment(Qt.AlignCenter)
        grid_frames.addWidget(self.cve_com_frame, 1, 2)

        self.vin_barra_frame = QLabel("Codigo vin barra")
        self.vin_barra_frame.setMaximumSize(250, 150)
        # self.vin_barra_frame.setStyleSheet("border: 2px solid gray;")
        self.vin_barra_frame.setStyleSheet("border: 2px solid #7f8c8d; background-color: #2c3e50; color: white; font-weight: bold;")
        self.vin_barra_frame.setAlignment(Qt.AlignCenter)
        grid_frames.addWidget(self.vin_barra_frame, 2, 2)

        layout.addLayout(grid_frames)


        grid_data = QGridLayout()

        self.vin_ult_title = QLabel("Data vin: ")
        self.vin_ult_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        grid_data.addWidget(self.vin_ult_title, 0, 0)
        
        self.vin_ult_label = QLabel("Esperando transferencia de datos...")
        self.vin_ult_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60;")
        grid_data.addWidget(self.vin_ult_label, 1, 0)

        self.datamatrix_link_title = QLabel("Data datamatrix link: ")
        self.datamatrix_link_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        grid_data.addWidget(self.datamatrix_link_title, 2, 0)
        
        self.datamatrix_link_label = QLabel("Esperando transferencia de datos...")
        self.datamatrix_link_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60;")
        grid_data.addWidget(self.datamatrix_link_label, 3, 0)

        self.datamatrix_num_title = QLabel("Data datamatrix num: ")
        self.datamatrix_num_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        grid_data.addWidget(self.datamatrix_num_title, 4, 0)
        
        self.datamatrix_num_label = QLabel("Esperando transferencia de datos...")
        self.datamatrix_num_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60;")
        grid_data.addWidget(self.datamatrix_num_label, 5, 0)

        self.pkn_largo_title = QLabel("Data pkn_largo: ")
        self.pkn_largo_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        grid_data.addWidget(self.pkn_largo_title, 0, 1)
        
        self.pkn_largo_label = QLabel("Esperando transferencia de datos...")
        self.pkn_largo_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60;")
        grid_data.addWidget(self.pkn_largo_label, 1, 1)

        self.cve_com_title = QLabel("Data cve com: ")
        self.cve_com_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        grid_data.addWidget(self.cve_com_title, 2, 1)
        
        self.cve_com_label = QLabel("Esperando transferencia de datos...")
        self.cve_com_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60;")
        grid_data.addWidget(self.cve_com_label, 3, 1)

        self.vin_barra_title = QLabel("Data vin barra: ")
        self.vin_barra_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        grid_data.addWidget(self.vin_barra_title, 4, 1)
        
        self.vin_barra_label = QLabel("Esperando transferencia de datos...")
        self.vin_barra_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60;")
        grid_data.addWidget(self.vin_barra_label, 5, 1)

        layout.addLayout(grid_data)

        layout_info = QVBoxLayout()

        self.tiempo_procesamiento_title = QLabel("Tiempo de procesamiento: ")
        self.tiempo_procesamiento_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")

        self.tiempo_procesamiento_label = QLabel("Esperando transferencia de datos...")
        self.tiempo_procesamiento_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60;")

        self.num_frame_title = QLabel("Numero de frames: ")
        self.num_frame_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")

        self.num_frame_label = QLabel("Esperando transferencia de datos...")
        self.num_frame_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60;")


        layout_info.addWidget(self.tiempo_procesamiento_title)
        layout_info.addWidget(self.tiempo_procesamiento_label)
        layout_info.addWidget(self.num_frame_title)
        layout_info.addWidget(self.num_frame_label)

        layout_info.addStretch()

        layout.addLayout(layout_info)

        self.btn_parar_analisis = QPushButton("Parar analisis")
        self.btn_parar_analisis.setStyleSheet("background-color: #b51212; font-weight: bold; font-size: 14px; height: 45px; color: white;")
        self.btn_parar_analisis.clicked.connect(self.parar_analisis)
        layout.addWidget(self.btn_parar_analisis)


        layout.addStretch()
        self.pagina_analisis.setLayout(layout)
        self.pages.addWidget(self.pagina_analisis)


        # Pagina 3 (Resultados)=========================================
        self.pagina_resultados = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("<h2>Resultados</h2>"))

        self.mapa_widget = MapaDashboard()
        layout.addWidget(self.mapa_widget)

        grid_btn = QGridLayout()

        self.btn_iniciar_procesamiento = QPushButton("Iniciar procesamiento")
        self.btn_iniciar_procesamiento.setStyleSheet("background-color: #2ecc71; font-weight: bold; font-size: 14px; height: 45px; color: white;")
        self.btn_iniciar_procesamiento.clicked.connect(self.iniciar_procesamiento_datos)
        grid_btn.addWidget(self.btn_iniciar_procesamiento, 0, 0)

        self.btn_actualizar_datos = QPushButton("Actualizar datos")
        self.btn_actualizar_datos.setStyleSheet("background-color: #2ecc71; font-weight: bold; font-size: 14px; height: 45px; color: white;")
        self.btn_actualizar_datos.clicked.connect(self.actualizar_datos)
        grid_btn.addWidget(self.btn_actualizar_datos, 0, 1)

        layout.addLayout(grid_btn)

        self.pagina_resultados.setLayout(layout)
        self.pages.addWidget(self.pagina_resultados)

        # ===============================================================

    def actualizar_video_seleccionado(self, ruta_archivo):
        """MÉTODO CORREGIDO: Guarda la ruta del archivo y actualiza la UI"""
        self.video_seleccionado = ruta_archivo
        nombre_corto = os.path.basename(ruta_archivo)
        
        # Modificar visualización del Label a éxito verde
        self.label_video.setText(f"<b>Video listo:</b> {nombre_corto}")
        self.label_video.setStyleSheet("color: #27ae60; font-size: 13px; padding: 5px; background-color: #f5fbf7; border: 1px solid #27ae60;")

    def actualizar_srt_seleccionado(self, ruta_archivo):
        self.srt_seleccionado = ruta_archivo
        nombre_corto = os.path.basename(ruta_archivo)
        
        # Modificar visualización del Label a éxito verde
        self.label_srt.setText(f"<b>SRT listo:</b> {nombre_corto}")
        self.label_srt.setStyleSheet("color: #27ae60; font-size: 13px; padding: 5px; background-color: #f5fbf7; border: 1px solid #27ae60;")


    def iniciar_analisis(self):
        if not self.video_seleccionado:
            QMessageBox.warning(self, "Falta archivo", "Por favor introduce un video primero.")
            return
        
        try:
            # Enviamos la ruta del archivo por HTTP POST a FastAPI
            res_video = requests.post(f"{self.api_url_analizador}/iniciar", json={"ruta_video": self.video_seleccionado})
            res_grafo = requests.post(f"{self.api_url_graficador}/iniciar", json={"ruta_waypoint": self.srt_seleccionado})
            if res_video.status_code == 200 and res_grafo.status_code == 200:
                self.pages.setCurrentIndex(1)
                self.timer_actualizador.start()
            else:
                raise requests.exceptions.RequestException
        except Exception:
            msg = QMessageBox(QMessageBox.Critical, "Conexión Fallida", "No se pudo comunicar con los servidores API. Verifica que FastAPI esté corriendo.", parent=self)
            msg.setStyleSheet("QLabel{ color: #c0392b; }")
            msg.exec()

        
    def parar_analisis(self):
        try:
            requests.post(f"{self.api_url_analizador}/parar")
            requests.post(f"{self.api_url_graficador}/parar")
        except Exception as e:
            print(f"Error al enviar parada: {e}")
        self.timer_actualizador.stop()
        self.pages.setCurrentIndex(0)


    def solicitar_actualizacion_servidor_analisis(self):
        """Pide el estado de procesamiento por HTTP GET y renderiza los bytes de imágenes"""
        try:
            res = requests.get(f"{self.api_url_analizador}/estado").json()
            
            # 1. Pintar textos detectados
            self.vin_ult_label.setText(res["vin_ult_data"])
            self.datamatrix_link_label.setText(res["datamatrix_link_data"])
            self.datamatrix_num_label.setText(res["datamatrix_num_data"])
            self.pkn_largo_label.setText(res["pkn_largo_data"])
            self.cve_com_label.setText(res["cve_com_data"])
            self.vin_barra_label.setText(res["vin_barra_data"])

            self.tiempo_procesamiento_label.setText(res["data_tiempo_procesamiento"])
            num_frame = res["data_num_frame"]
            num_frame_max = res["data_num_frame_max"]
            progreso = str(f"{num_frame}/{num_frame_max}")
            self.num_frame_label.setText(progreso)

            # 2. Reconvertir e inyectar el video principal (Base64 -> QPixmap)
            if res["frame"]:
                self.convertir_b64_a_label(res["frame"], self.main_frame)
            # if res["recorte_etiqueta"]:
            #     self.convertir_b64_a_label(res["recorte_etiqueta"], self.label_frame)
            if res["recorte_vin_ult"]:
                self.convertir_b64_a_label(res["recorte_vin_ult"], self.vin_ult_frame)
            if res["recorte_datamatrix_link"]:
                self.convertir_b64_a_label(res["recorte_datamatrix_link"], self.datamatrix_link_frame)
            if res["recorte_datamatrix_num"]:
                self.convertir_b64_a_label(res["recorte_datamatrix_num"], self.datamatrix_num_frame)
            if res["recorte_pkn_largo"]:
                self.convertir_b64_a_label(res["recorte_pkn_largo"], self.pkn_largo_frame)
            if res["recorte_cve_com"]:
                self.convertir_b64_a_label(res["recorte_cve_com"], self.cve_com_frame)
            if res["recorte_vin_barra"]:
                self.convertir_b64_a_label(res["recorte_vin_barra"], self.vin_barra_frame)

            # Si el backend avisa que ya acabó, paramos el timer de peticiones
            if not res["corriendo"] and res["frame_actual_base64"] is not None:
                self.timer_actualizador.stop()
                QMessageBox.information(self, "Terminado", "El análisis del video concluyó exitosamente.")
                
        except Exception as e:
            self.timer_actualizador.stop()
            print(f"Error al conectar con la API: {e}")

    def solicitar_actualizacion_servidor_grafica(self):
        """Pide el estado de procesamiento por HTTP GET y renderiza los bytes de imágenes"""
        try:
            res = requests.get(f"{self.api_url_graficador}/estado").json()
            
            if res["grafica"]:
                self.convertir_b64_a_label(res["grafica"], self.graph_frame)

            if not res["corriendo"] and res["frame_actual_base64"] is not None:
                self.timer_actualizador.stop()
                QMessageBox.information(self, "Terminado", "El análisis del video concluyó exitosamente.")
                
        except Exception as e:
            self.timer_actualizador.stop()
            print(f"Error al conectar con la API: {e}")

    def convertir_b64_a_label(self, base64_str, label_target):
        img_data = base64.b64decode(base64_str)
        qimage = QImage.fromData(img_data)
        pixmap = QPixmap.fromImage(qimage)
        label_target.setPixmap(pixmap.scaled(label_target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def cambiar_a_pagina_resultados(self):
        self.mapa_widget.cargar_mapa()
        self.pages.setCurrentIndex(2)

    def iniciar_procesamiento_datos(self):
        res_procesamiento = requests.post(f"{self.api_url_procesador}/iniciar", json={"iniciar": int(1)})

    def actualizar_datos(self):
        self.mapa_widget.cargar_mapa()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = DroneDashboard()
    window.show()

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\nInterrupción por teclado")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Programa finalizado")