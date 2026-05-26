import sys
import os
import sqlite3
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView

# --- CONFIGURACIÓN DE LA BASE DE DATOS ---
# Usamos el nombre del script anterior: imagenes_dji.db
DB_PATH = "imagenes_dji.db"

class MapDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DJI QR & GPS Dashboard")
        self.resize(1000, 700)

    def initUI(self):
        # Contenedor principal
        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Crear el visor web
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)

        # Cargar el mapa
        self.load_map()

    def get_data_from_db(self):
        """Extrae los puntos de la base de datos para pasarlos a JavaScript."""
        detecciones = []
        if not os.path.exists(DB_PATH):
            return detecciones

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            # Ajustado a la tabla 'fotos' del script anterior
            cursor.execute("SELECT nombre_archivo, qr_contenido, latitud, longitud FROM fotos")
            rows = cursor.fetchall()
            for r in rows:
                # Filtrar puntos que no tengan GPS (None)
                if r[2] and r[3]:
                    detecciones.append({
                        "name": r[0],
                        "qr": r[1],
                        "lat": r[2],
                        "lon": r[3]
                    })
            conn.close()
        except Exception as e:
            print(f"Error DB: {e}")
        return detecciones

    def load_map(self):
        """Genera un HTML con Leaflet e inyecta los datos de la DB."""
        puntos = self.get_data_from_db()
        # Convertimos la lista de Python a un string JSON para JavaScript
        puntos_js = json.dumps(puntos)

        # Definimos el punto central del mapa (el primer punto o 0,0)
        center_lat = puntos[0]['lat'] if puntos else 0
        center_lon = puntos[0]['lon'] if puntos else 0

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <style>
                #map {{ height: 100vh; width: 100%; margin: 0; padding: 0; }}
                body {{ margin: 0; }}
                .popup-custom {{ font-family: sans-serif; }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                // Inicializar el mapa
                var map = L.map('map').setView([{center_lat}, {center_lon}], 13);

                // Capa de OpenStreetMap
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '© OpenStreetMap'
                }}).addTo(map);

                // Datos inyectados desde Python
                var puntos = {puntos_js};

                // Añadir marcadores
                puntos.forEach(function(p) {{
                    var marker = L.marker([p.lat, p.lon]).addTo(map);
                    marker.bindPopup(`
                        <div class="popup-custom">
                            <b>Imagen:</b> ${{p.name}}<br>
                            <b>QR:</b> <code style="color:blue;">${{p.qr}}</code><br>
                            <b>Coordenadas:</b> ${{p.lat.toFixed(6)}}, ${{p.lon.toFixed(6)}}
                        </div>
                    `);
                }});

                // Si hay puntos, ajustar el zoom para verlos todos
                if (puntos.length > 0) {{
                    var group = new L.featureGroup(puntos.map(p => L.marker([p.lat, p.lon])));
                    map.fitBounds(group.getBounds());
                }}
            </script>
        </body>
        </html>
        """
        self.browser.setHtml(html_content)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MapDashboard()
    window.initUI()
    window.show()
    sys.exit(app.exec_())