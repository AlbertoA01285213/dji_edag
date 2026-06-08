#!/usr/bin/env bash

# Configuración de rutas
PATH_PROYECTO="/home/alberto/Documents/dji_edag/app"
API_YOLO="analizador.py"
API_BD="procesador.py"
API_DRONE="graficador.py"
FRONTEND="app_2.py"

echo "============================================="
echo "   Iniciando Sistema de Análisis DJI EDAG    "
echo "============================================="

# Función para apagar TODO de golpe cuando cierres la app
interrupcion_limpia() {
    echo -e "\n[Cerrando] Apagando todos los servicios en segundo plano..."
    kill $(jobs -p) 2>/dev/null
    kill -9 -$GRUPO_PROCESOS 2>/dev/null
    exit 0
}

# Registrar la función de limpieza para cuando se presione Ctrl+C o termine el script
trap interrupcion_limpia SIGINT SIGTERM EXIT

cd "$PATH_PROYECTO" || exit 1

echo "[1/4] Lanzando Servidor YOLO..."
python3 "$API_YOLO" &

echo "[2/4] Lanzando Servidor de Datos..."
python3 "$API_BD" &

echo "[3/4] Lanzando Servidor de Telemetría..."
python3 "$API_DRONE" &

echo "Esperando a que las APIs estén listas..."
sleep 3

echo "[4/4] Abriendo Interfaz Gráfica PySide6..."
python3 "$FRONTEND"

interrupcion_limpia