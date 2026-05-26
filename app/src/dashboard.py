import sys
import os
import sqlite3
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView

# --- CONFIGURACIÓN DE LA BASE DE DATOS ---
DB_PATH = "imagenes_dji.db"

class MapDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DJI QR & GPS Dashboard")
        self.resize(1000, 700)
        self.initUI()

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
            cursor.execute("SELECT nombre_archivo, qr_contenido, latitud, longitud FROM fotos")
            rows = cursor.fetchall()
            for r in rows:
                # Filtrar puntos que no tengan GPS (None)
                if r[2] and r[3]:
                    detecciones.append({
                        "name": r[0],
                        "qr": r[1] if r[1] else "Sin QR",
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
                .popup-custom {{ 
                    font-family: sans-serif; 
                    font-size: 12px;
                }}
                .info-tooltip {{
                    background: white;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    padding: 10px;
                    font-family: sans-serif;
                    font-size: 12px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                }}
                .info-tooltip b {{
                    color: #2c3e50;
                }}
                .info-tooltip code {{
                    background: #f0f0f0;
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-size: 11px;
                }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                // Inicializar el mapa
                var map = L.map('map').setView([{center_lat}, {center_lon}], 13);

                // SOLUCIÓN PROBLEMA 1: Usar un tile provider que no requiera Referer
                // Opción 1: CartoDB (recomendada, bonito y confiable)
                L.tileLayer('https://cartodb-basemaps-{{s}}.global.ssl.fastly.net/light_all/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
                    subdomains: 'abcd',
                    maxZoom: 19
                }}).addTo(map);

                // Opción 2: Stamen (alternativa, tonos más cálidos)
                // L.tileLayer('https://stamen-tiles-{{s}}.a.ssl.fastly.net/toner/{{z}}/{{x}}/{{y}}.png', {{
                //     attribution: 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                //     subdomains: 'abcd',
                //     minZoom: 0,
                //     maxZoom: 20
                // }}).addTo(map);

                // Datos inyectados desde Python
                var puntos = {puntos_js};

                // SOLUCIÓN PROBLEMA 2: Tooltip en hover en lugar de popup
                puntos.forEach(function(p) {{
                    var marker = L.marker([p.lat, p.lon]).addTo(map);
                    
                    // Contenido del tooltip (aparece al hacer hover)
                    var tooltipContent = `
                        <div class="info-tooltip">
                            <b>📷 Imagen:</b> ${{p.name}}<br>
                            <b>🔲 QR:</b> <code>${{p.qr}}</code><br>
                            <b>📍 Coordenadas:</b><br>
                            &nbsp;&nbsp;Lat: ${{p.lat.toFixed(6)}}<br>
                            &nbsp;&nbsp;Lon: ${{p.lon.toFixed(6)}}
                        </div>
                    `;
                    
                    // Bind tooltip (aparece al hacer hover)
                    marker.bindTooltip(tooltipContent, {{
                        permanent: false,     // No permanente
                        direction: 'top',     // Aparece arriba del marcador
                        offset: [0, -20],     // Offset para que no tape el marcador
                        opacity: 0.95,
                        sticky: true          // Sigue al mouse
                    }});
                    
                    // Opcional: También mantener popup para click (por si prefieren)
                    marker.bindPopup(`
                        <div class="popup-custom">
                            <b>Imagen:</b> ${{p.name}}<br>
                            <b>QR:</b> <code>${{p.qr}}</code><br>
                            <b>Coordenadas:</b><br>
                            ${{p.lat.toFixed(6)}}, ${{p.lon.toFixed(6)}}
                        </div>
                    `);
                }});

                // Si hay puntos, ajustar el zoom para verlos todos
                if (puntos.length > 0) {{
                    var group = new L.featureGroup(puntos.map(p => L.marker([p.lat, p.lon])));
                    map.fitBounds(group.getBounds());
                }}
                
                // Añadir contador de puntos al mapa
                var infoControl = L.control({{position: 'bottomright'}});
                infoControl.onAdd = function(map) {{
                    var div = L.DomUtil.create('div', 'info');
                    div.style.backgroundColor = 'white';
                    div.style.padding = '10px';
                    div.style.borderRadius = '5px';
                    div.style.boxShadow = '0 1px 5px rgba(0,0,0,0.2)';
                    div.innerHTML = '<b>Total puntos:</b> ' + puntos.length;
                    return div;
                }};
                infoControl.addTo(map);
            </script>
        </body>
        </html>
        """
        self.browser.setHtml(html_content)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MapDashboard()
    window.show()
    sys.exit(app.exec_())