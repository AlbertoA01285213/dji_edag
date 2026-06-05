### App
Un problema para las ensambladoras son sus parques vehiculares. Debido a la magnitud es dificil llevar un inventario de donde estan y a donde van los carros. Para evitar ese problema, se desarrollo esta aplicacion. Utilizando un dron DJI, se disena una ruta utilizando cualquier aplicacion. El dron automaticamente volara y grabara todo el recorrido. 

*Insertar gif del dron volando*

Una vez conseguido el video y el archivo de telemetria, estos se meten en la aplicacion la cual ejecutara el analisis del video y guardara los datos obtenidos en una base de datos para su posterior analisis.

*Insertar gif de la aplicacion analizando* 

## Como funciona
El programa se basa en 1 codigo principal el cual es el analizador. Este es el encargado de recibir cada imagen del video, analizarla y dar los resultados. El scipt agarra la primera foto. Usando YOLO la analiza foto en busqueda de la etiqueta. Una vez encontrada la etiquta, la recorta y ese recorte lo vuelve a analizar con YOLO para buscar los codigos de barra y recorta cada uno de los codigos encontrados. Utilizando zxingcpp se analiza cada codigo para obtener su informacion.
A la vez, esta corriendo el codigo del dashboard. El dashboard funciona como el cliente. Este le indica al analizador el request de datos. El analziador recibe el request y le manda los datos y las imagenes. Posteriormente el dashboard recibe los datos y los muestra.
EL codigo del graficador es meramente estetico. Este se encarga de leer el archivo de la telemetria, del archivo obtiene los datos de posicion del dron. Luego, utiliza esos datos para generar un path visual del dron y utiliza la informacion del frame del video para sincronizar y crear una animacion de la ruta del dron conforme analiza las imagenes.

## Como correrlo
Para correr el repositorio, primero debes de clonarlo.

```
git clone ...
```

Tambien ocuparas las librerias
- pyside
- uvicorn
- fast api
- no se que mas

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