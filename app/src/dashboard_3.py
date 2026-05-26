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
        self.setWindowTitle("DJI QR & GPS Dashboard - Vista Satelital")
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
        """Genera un HTML con Leaflet y mapa satelital de ESRI."""
        puntos = self.get_data_from_db()
        puntos_js = json.dumps(puntos)

        # Centro del mapa (primer punto o coordenadas por defecto)
        center_lat = puntos[0]['lat'] if puntos else 20.5
        center_lon = puntos[0]['lon'] if puntos else -100.5

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
                .control-buttons {{
                    background: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 1px 5px rgba(0,0,0,0.2);
                    margin-bottom: 10px;
                }}
                .control-buttons button {{
                    margin: 2px;
                    padding: 5px 10px;
                    cursor: pointer;
                }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                // Inicializar el mapa
                var map = L.map('map').setView([{center_lat}, {center_lon}], 15);

                // ============================================
                // MAPA SATELITAL DE ESRI (World Imagery)
                // ============================================
                // Esta capa NO requiere API key y tiene imágenes de hasta 0.3m de resolución
                // Fuente: ESRI, Maxar (antes DigitalGlobe), Airbus, y comunidades GIS
                var satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
                    attribution: '&copy; <a href="https://www.esri.com">Esri</a> | &copy; <a href="https://www.maxar.com/">Maxar</a> | Earthstar Geographics',
                    maxZoom: 20,
                    subdomains: ['services']
                }}).addTo(map);

                // Opcional: Capa de etiquetas para el mapa satelital
                // (útil para identificar calles y lugares)
                var labelsLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
                    attribution: '&copy; Esri',
                    maxZoom: 20
                }});
                
                // Control para alternar entre satelital y mapa base
                var baseMaps = {{
                    "🛰️ Satelital (ESRI)": satelliteLayer,
                    "🗺️ Calles (OSM)": L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                        attribution: '&copy; OpenStreetMap'
                    }})
                }};
                
                var overlayMaps = {{
                    "🏷️ Etiquetas": labelsLayer
                }};
                
                L.control.layers(baseMaps, overlayMaps, {{position: 'topright'}}).addTo(map);
                
                // Datos inyectados desde Python
                var puntos = {puntos_js};

                // Añadir marcadores con tooltip hover
                puntos.forEach(function(p) {{
                    var marker = L.marker([p.lat, p.lon]).addTo(map);
                    
                    var tooltipContent = `
                        <div class="info-tooltip">
                            <b>📷 Imagen:</b> ${{p.name}}<br>
                            <b>🔲 QR:</b> <code>${{p.qr}}</code><br>
                            <b>📍 Coordenadas:</b><br>
                            &nbsp;&nbsp;Lat: ${{p.lat.toFixed(6)}}<br>
                            &nbsp;&nbsp;Lon: ${{p.lon.toFixed(6)}}
                        </div>
                    `;
                    
                    marker.bindTooltip(tooltipContent, {{
                        permanent: false,
                        direction: 'top',
                        offset: [0, -20],
                        opacity: 0.95,
                        sticky: true
                    }});
                    
                    marker.bindPopup(`
                        <div class="popup-custom">
                            <b>Imagen:</b> ${{p.name}}<br>
                            <b>QR:</b> <code>${{p.qr}}</code><br>
                            <b>Coordenadas:</b><br>
                            ${{p.lat.toFixed(6)}}, ${{p.lon.toFixed(6)}}
                        </div>
                    `);
                }});

                // Ajustar zoom para ver todos los puntos
                if (puntos.length > 0) {{
                    var group = new L.featureGroup(puntos.map(p => L.marker([p.lat, p.lon])));
                    map.fitBounds(group.getBounds());
                }}
                
                // Contador de puntos
                var infoControl = L.control({{position: 'bottomright'}});
                infoControl.onAdd = function(map) {{
                    var div = L.DomUtil.create('div', 'info');
                    div.style.backgroundColor = 'white';
                    div.style.padding = '10px';
                    div.style.borderRadius = '5px';
                    div.style.boxShadow = '0 1px 5px rgba(0,0,0,0.2)';
                    div.innerHTML = '<b>🛰️ Total puntos:</b> ' + puntos.length;
                    return div;
                }};
                infoControl.addTo(map);
                
                // Escala del mapa
                L.control.scale({{metric: true, imperial: false}}).addTo(map);
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