### App
Un problema para las ensambladoras son sus parques vehiculares. Debido a la magnitud es dificil llevar un inventario de donde estan y a donde van los carros. Para evitar ese problema, se desarrollo esta aplicacion. Utilizando un dron DJI, se disena una ruta utilizando cualquier aplicacion. El dron automaticamente volara y grabara todo el recorrido. 

*Insertar gif del dron volando*

Una vez conseguido el video y el archivo de telemetria, estos se meten en la aplicacion la cual ejecutara el analisis del video y guardara los datos obtenidos en una base de datos para su posterior analisis.

*Insertar gif de la aplicacion analizando* 

## Como funciona
El programa se basa en 1 codigo principal el cual es el analizador. Este es el encargado de recibir cada imagen del video, analizarla y dar los resultados. El script agarra la primera foto. Usando YOLO, analiza la foto en busqueda de la etiqueta. Una vez encontrada la etiqueta, se realiza el recorte y vuelve a analizar con YOLO para buscar las diferentes secciones de la etiqueta. Utilizando zxingcpp se analizan los Code39 y Datamatrix y con PaddleOCR se analiza los últimos dígitos del VIN.

A la vez, esta corriendo el codigo del dashboard. El dashboard funciona como el cliente. Este le indica al analizador el request de datos. El analizador recibe el request y le manda los datos y las imagenes. Posteriormente el dashboard recibe los datos y los muestra.
El código del graficador es meramente estético. Este se encarga de leer el archivo de la telemetría, del archivo obtiene los datos de posición del dron. Luego, utiliza esos datos para generar un path visual del dron y utiliza la información del frame del video para sincronizar y crear una animación de la ruta del dron conforme analiza las imagenes.

## Como correrlo
Para correr el repositorio, primero se necesita clonar.

```
git clone ...
```

Tambien ocuparas las librerias
- PySide6
- requests
- folium
- opencv-python
- numpy
- ultralytics
- zxing-cpp
- pylibdtmx
- paddleocr
- fastapi
- uvicorn
- matplotlib

Las librerías se encuentran en requirements.txt.

Correrlo es sencillo. En la primera terminal corres la aplicacion.
```
python3 app.py
```

En la segunda terminal corres el analizador del video.
```
python3 analizador.py
```

En la tercera y ultima terminal corres el graficador.
```
python3 graficador.py
```