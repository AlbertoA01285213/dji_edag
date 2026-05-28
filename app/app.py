#!/usr/bin/env python3
import os
import sys
import cv2
import base64
import sqlite3
import requests
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, NavSatFix  # o el tipo que uses para odometría
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QCursor
from PySide6.QtCore import Signal, QThread, Qt, Slot, QPoint, QObject, QTimer
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QToolTip, QComboBox, 
                             QVBoxLayout, QHBoxLayout, QWidget, QStackedWidget, QMessageBox,
                             QLabel, QFrame, QGridLayout, QSpinBox, QDoubleSpinBox, QWidget)

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from example_interfaces.srv import SetBool

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
        # Estilo visual de caja gris con bordes punteados (estilo dropzone)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #3498db;
                border-radius: 8px;
                background-color: #ecf0f1;
            }
            QFrame:hover {
                background-color: #e8f4f8;
                border: 2px dashed #2980b9;
            }
        """)
        self.setMinimumHeight(150)
        self.setAcceptDrops(True) # ¡CRÍTICO para permitir Drag & Drop!

        # Texto interno instructivo
        layout = QVBoxLayout()
        self.label_info = QLabel("Arrastra y suelta tu archivo de video aquí\n— o —")
        self.label_info.setAlignment(Qt.AlignCenter)
        self.label_info.setStyleSheet("color: #7f8c8d; font-size: 14px; font-weight: bold; border: none; background: transparent;")
        layout.addWidget(self.label_info)
        self.setLayout(layout)

    # Evento cuando el mouse entra arrastrando algo
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            # Verificar que sea un archivo local
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile():
                event.acceptProposedAction() # Aceptar la acción visual de copia

    # Evento cuando el operador suelta el archivo definitivo
    def dropEvent(self, event):
        for url in event.mimeData().urls():
            ruta_archivo = url.toLocalFile()
            # Validar de forma sencilla que tenga extensión de video común
            if ruta_archivo.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                self.callback_archivo(ruta_archivo)
                event.acceptProposedAction()
            else:
                QMessageBox.warning(self, "Archivo no válido", "Por favor, arrastra solo archivos de video (.mp4, .avi, .mkv, .mov).")


class DroneDashboard(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Drone Mission Control v1.0")
        self.resize(1000, 650)

        self.video_seleccionado = None

        self.api_url = "http://127.0.0.1:8000"

        self.timer_actualizador = QTimer()
        self.timer_actualizador.setInterval(60) # Actualiza a ~16 FPS la GUI
        self.timer_actualizador.timeout.connect(self.solicitar_actualizacion_servidor)        

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
        self.btn_resultados.clicked.connect(lambda: self.pages.setCurrentIndex(2))

    
    def init_pages(self):
        # =========================================================================
        # PAGINA 1: Configuración de análisis
        # =========================================================================
        self.pagina_configuracion = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("<h2>Seleccion de datos</h2>"))

        layout.addWidget(QLabel("Selecciona el video"))
        layout.addSpacing(15)
        self.zona_drag_drop = ZonaArrastrarVideo(self.actualizar_video_seleccionado)
        layout.addWidget(self.zona_drag_drop)
        layout.addSpacing(15)

        self.label_video = QLabel("<b>Video seleccionado:</b> Ninguno (Usa el buscador o arrastra un archivo)")
        self.label_video.setStyleSheet("color: #c0392b; font-size: 13px; padding: 5px; background-color: #fafdff; border: 1px solid #dcdde1;")
        layout.addWidget(self.label_video)

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

        grid = QGridLayout()

        self.main_frame = QLabel("Esperando video")
        self.main_frame.setMaximumSize(420, 230) # 640 360
        # self.main_frame.setStyleSheet("border: 2px solid gray;")
        self.main_frame.setStyleSheet("border: 2px solid #7f8c8d; background-color: #2c3e50; color: white; font-weight: bold;")
        self.main_frame.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.main_frame, 0, 0, 2, 1)

        self.barcode_frame = QLabel("Codigo de barras")
        self.barcode_frame.setMaximumSize(250, 150)
        # self.barcode_frame.setStyleSheet("border: 2px solid gray;")
        self.barcode_frame.setStyleSheet("border: 2px solid #7f8c8d; background-color: #2c3e50; color: white; font-weight: bold;")
        self.barcode_frame.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.barcode_frame, 0, 1)

        self.qrcode_frame = QLabel("Codigo QR")
        self.qrcode_frame.setMaximumSize(250, 150)
        # self.qrcode_frame.setStyleSheet("border: 2px solid gray;")
        self.qrcode_frame.setStyleSheet("border: 2px solid #7f8c8d; background-color: #2c3e50; color: white; font-weight: bold;")
        self.qrcode_frame.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.qrcode_frame, 1, 1)

        self.vin_frame = QLabel("Codigo VIN")
        self.vin_frame.setMaximumSize(250, 150)
        # self.vin_frame.setStyleSheet("border: 2px solid gray;")
        self.vin_frame.setStyleSheet("border: 2px solid #7f8c8d; background-color: #2c3e50; color: white; font-weight: bold;")
        self.vin_frame.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.vin_frame, 2, 1)

        self.datamatrix_frame = QLabel("Codigo DataMatrix")
        self.datamatrix_frame.setMaximumSize(250, 150)
        # self.datamatrix_frame.setStyleSheet("border: 2px solid gray;")
        self.datamatrix_frame.setStyleSheet("border: 2px solid #7f8c8d; background-color: #2c3e50; color: white; font-weight: bold;")
        self.datamatrix_frame.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.datamatrix_frame, 3, 1)

        self.label_frame = QLabel("Codigo label")
        self.label_frame.setMaximumSize(280, 165)
        # self.label_frame.setStyleSheet("border: 2px solid gray;")
        self.label_frame.setStyleSheet("border: 2px solid #7f8c8d; background-color: #2c3e50; color: white; font-weight: bold;")
        self.label_frame.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.label_frame, 3, 0, 0, 0)

        layout.addLayout(grid)

        layout_info = QVBoxLayout()
        self.barcode_title = QLabel("Codigo de barras: ")
        self.barcode_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")

        self.barcode_label = QLabel("Esperando transferencia de datos...")
        self.barcode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60;")

        self.qrcode_title = QLabel("Codigo QR: ")
        self.qrcode_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")

        self.qrcode_label = QLabel("Esperando transferencia de datos...")
        self.qrcode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60;")

        self.vin_title = QLabel("Codigo VIN: ")
        self.vin_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")

        self.vin_label = QLabel("Esperando transferencia de datos...")
        self.vin_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60;")

        self.datamatrix_title = QLabel("Codigo DataMatrix: ")
        self.datamatrix_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")

        self.datamatrix_label = QLabel("Esperando transferencia de datos...")
        self.datamatrix_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60;")


        self.tiempo_procesamiento_title = QLabel("Tiempo de procesamiento: ")
        self.tiempo_procesamiento_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")

        self.tiempo_procesamiento_label = QLabel("Esperando transferencia de datos...")
        self.tiempo_procesamiento_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60;")

        self.num_frame_title = QLabel("Numero de frames: ")
        self.num_frame_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")

        self.num_frame_label = QLabel("Esperando transferencia de datos...")
        self.num_frame_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60;")


        layout_info.addWidget(self.barcode_title)
        layout_info.addWidget(self.barcode_label)
        layout_info.addWidget(self.qrcode_title)
        layout_info.addWidget(self.qrcode_label)
        layout_info.addWidget(self.vin_title)
        layout_info.addWidget(self.vin_label)
        layout_info.addWidget(self.datamatrix_title)
        layout_info.addWidget(self.datamatrix_label)

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
        self.page_config = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("<h2>Resultados</h2>"))

        # ===============================================================

    def actualizar_video_seleccionado(self, ruta_archivo):
        """MÉTODO CORREGIDO: Guarda la ruta del archivo y actualiza la UI"""
        self.video_seleccionado = ruta_archivo
        nombre_corto = os.path.basename(ruta_archivo)
        
        # Modificar visualización del Label a éxito verde
        self.label_video.setText(f"<b>Video listo:</b> {nombre_corto}")
        self.label_video.setStyleSheet("color: #27ae60; font-size: 13px; padding: 5px; background-color: #f5fbf7; border: 1px solid #27ae60;")


    def iniciar_analisis(self):
        if not self.video_seleccionado:
            QMessageBox.warning(self, "Falta archivo", "Por favor introduce un video primero.")
            return
        
        try:
            # Enviamos la ruta del archivo por HTTP POST a FastAPI
            respuesta = requests.post(f"{self.api_url}/iniciar", json={"ruta_video": self.video_seleccionado})
            if respuesta.status_code == 200:
                self.pages.setCurrentIndex(1)
                # Arrancar el timer para pedir datos constantemente
                self.timer_actualizador.start()
            else:
                QMessageBox.critical(self, "Error Servidor", "El servidor rechazó la solicitud.")
        except requests.exceptions.ConnectionError:
            QMessageBox.critical(self, "Servidor Apagado", "Error: Recuerda encender primero 'analisis_server.py' en una terminal.")

    def parar_analisis(self):
        respuesta = requests.post(f"{self.api_url}/parar")
        if respuesta.status_code == 200:
            self.pages.setCurrentIndex(2)
            self.timer_actualizador.stop()

        else:
            QMessageBox.critical(self, "Error Servidor", "El servidor rechazó la solicitud.")

        
    def solicitar_actualizacion_servidor(self):
        """Pide el estado de procesamiento por HTTP GET y renderiza los bytes de imágenes"""
        try:
            res = requests.get(f"{self.api_url}/estado").json()
            
            # 1. Pintar textos detectados
            self.barcode_label.setText(res["data_barcode"])
            self.qrcode_label.setText(res["data_qr"])
            self.vin_label.setText(res["data_vin"])
            self.datamatrix_label.setText(res["data_datamax"])

            self.tiempo_procesamiento_label.setText(res["data_datamax"])
            num_frame = res["data_num_frame"]
            num_frame_max = res["data_num_frame_max"]
            progreso = str(f"{num_frame}/{num_frame_max}")
            self.datamatrix_label.setText(res["data_datamax"])
            self.num_frame_label.setText(progreso)

            # 2. Reconvertir e inyectar el video principal (Base64 -> QPixmap)
            if res["frame"]:
                self.convertir_b64_a_label(res["frame"], self.main_frame)
            if res["recorte_etiqueta"]:
                self.convertir_b64_a_label(res["recorte_etiqueta"], self.label_frame)
            if res["recorte_barcode"]:
                self.convertir_b64_a_label(res["recorte_barcode"], self.barcode_frame)
            if res["recorte_qr"]:
                self.convertir_b64_a_label(res["recorte_qr"], self.qrcode_frame)
            if res["recorte_vin"]:
                self.convertir_b64_a_label(res["recorte_vin"], self.vin_frame)
            if res["recorte_datamax"]:
                self.convertir_b64_a_label(res["recorte_datamax"], self.datamatrix_frame)

            # Si el backend avisa que ya acabó, paramos el timer de peticiones
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