# Entendiendo_una_Red_Neuronal

Aplicacion web educativa construida con Flask y JavaScript para visualizar, entrenar y entender una red neuronal de forma interactiva. El objetivo del proyecto es explicar el comportamiento interno de una red sencilla con una interfaz visual clara, no competir con frameworks de entrenamiento ni exponer un backend de machine learning complejo.

## Vision general

La aplicacion actual se centra en un caso didactico muy concreto: una red de regresion que aprende la conversion de grados Celsius a Fahrenheit.

Ese enfoque tiene una ventaja fuerte: al usar un problema simple y controlado, es facil ver con claridad:

- como entra el dato
- como se transforma capa a capa
- como afectan los pesos y sesgos
- como cambia la prediccion tras entrenar
- como evoluciona la perdida

El proyecto esta planteado como una herramienta para aprender, experimentar y observar.

## Que hace la aplicacion hoy

La experiencia activa que sirve Flask ofrece:

- un panel lateral para configurar la red
- un lienzo SVG que dibuja nodos y conexiones
- una grafica flotante de entrenamiento
- una tarjeta didactica con explicaciones contextuales
- una tarjeta para previsualizar la funcion de activacion

La persona usuaria puede:

- introducir un valor en Celsius
- elegir entre 1 y 4 capas ocultas
- definir entre 1 y 5 neuronas por capa
- seleccionar `sigmoid`, `relu` o `tanh`
- ejecutar un `Forward` manual
- entrenar por tandas
- reiniciar los pesos
- mostrar u ocultar pesos, grafica y tarjeta formativa

## Arquitectura tecnica

La arquitectura es intencionalmente ligera:

- Flask sirve la pagina principal, los recursos estaticos y un endpoint de salud.
- El modelo se simula y entrena en el navegador.
- TensorFlow.js se carga desde CDN.
- La visualizacion se renderiza con SVG y JavaScript vanilla.

Esto significa que el backend no entrena nada. Python se usa como contenedor web minimo y toda la logica de aprendizaje visible ocurre en cliente.

## Flujo funcional real

1. Flask entrega `templates/main.html`.
2. La pagina carga estilos propios y el script `assets/js/node_canvas.js`.
3. El navegador carga TensorFlow.js desde `cdn.jsdelivr.net`.
4. `node_canvas.js` inicializa una red neuronal simple con pesos aleatorios.
5. El usuario modifica parametros y lanza `Forward` o `Entrenar`.
6. El script recalcula activaciones, salida, perdida y metricas.
7. El mismo script redibuja la red, las conexiones y la grafica de progreso.

## Stack usado

- Python 3
- Flask
- HTML + Jinja
- CSS propio
- JavaScript vanilla
- TensorFlow.js
- SVG

## Estructura del repositorio

```text
Entendiendo_una_Red_Neuronal/
|-- Entendiendo_una_Red_Neuronal.py
|-- config.py
|-- requirements.txt
|-- README.md
|-- LICENSE
|-- app/
|   |-- api/
|   |   `-- __init__.py
|   |-- db/
|   |   `-- __init__.py
|   `-- educational_content.py
|-- assets/
|   |-- css/
|   |   |-- base.css
|   |   |-- emesa-header.css
|   |   |-- main.css
|   |   `-- componentes/
|   |       `-- EncabezadoAplicacion.css
|   |-- img/
|   |   |-- favicon.png
|   |   |-- Logo_EMESA.png
|   |   `-- PNGs/
|   |       `-- Logo_EMESA.png
|   `-- js/
|       |-- node_canvas.js
|       `-- learning_app.js
|-- templates/
|   |-- login.html
|   |-- main.html
|   `-- partials/
|       `-- header.html
|-- data/
|   |-- celsius_fahrenheit/
|   |   |-- train_100.csv
|   |   `-- test_25.csv
|   `-- skill_test_favicon_01/
|       `-- Demo_Favicon/
|-- docs/
|   |-- 00_introduccion.md
|   |-- 01_neurona_y_capas.md
|   |-- 02_forward_pass.md
|   |-- 03_activaciones.md
|   |-- 04_perdida_y_optimizacion.md
|   |-- 05_backprop_gradiente.md
|   |-- 06_como_elegir_arquitectura.md
|   |-- 07_casos_practicos.md
|   `-- glosario.md
`-- doc/
    |-- explicacion_red_neuronal_muy_basica.md
    `-- explicacion_red_neuronal_ampliada.md
```

## Archivos clave

### `Entendiendo_una_Red_Neuronal.py`

Es el punto de entrada de la aplicacion.

Responsabilidades:

- crear la app Flask
- servir `/`
- exponer `GET /api/health`
- devolver el favicon
- arrancar usando `PORT` desde `config.py`

### `templates/main.html`

Define la interfaz activa del simulador.

Incluye:

- cabecera
- panel de control
- contenedor del lienzo de red
- zona de grafica
- panel didactico
- tarjeta de activacion

### `assets/js/node_canvas.js`

Es el nucleo funcional real del proyecto actual. Concentra casi toda la logica:

- genera los datasets sinteticos de train y test
- inicializa pesos y sesgos
- calcula el forward pass
- entrena muestra a muestra
- evalua metricas
- dibuja nodos, aristas y tooltips
- gestiona animaciones y eventos de interfaz

En la practica, este archivo implementa tanto el simulador como la visualizacion.

### `app/educational_content.py`

Guarda una base de contenido didactico mas ambiciosa que la version hoy publicada:

- catalogo amplio de activaciones
- multiples casos de uso (regresion, binaria, multiclase, XOR)
- escenarios de gradiente
- errores comunes
- fases de implementacion
- wireframes y microcopy

Es importante entender que este archivo no esta conectado de forma activa a la `main.html` actual. Representa una base de contenido preparada para una futura expansion del proyecto.

### `assets/js/learning_app.js`

Tambien apunta a una evolucion futura:

- contiene logica para una app educativa mas grande
- soporta mas ejemplos y paneles
- trabaja con el contenido de `app/educational_content.py`

Sin embargo, `templates/main.html` no lo carga ahora mismo, asi que no forma parte de la experiencia que se sirve hoy. Es codigo adelantado o en preparacion.

### `docs/` y `doc/`

El proyecto incluye bastante material teorico en Markdown:

- `docs/` divide el contenido en modulos tematicos
- `doc/explicacion_red_neuronal_muy_basica.md` ofrece una version inicial para principiantes
- `doc/explicacion_red_neuronal_ampliada.md` desarrolla una explicacion extensa y detallada

Esto refuerza el enfoque formativo del repositorio: no solo hay demo visual, tambien hay base teorica escrita.

## Dataset y enfoque didactico

El caso activo usa una relacion matematica conocida:

`F = C * 9 / 5 + 32`

En la interfaz actual:

- el conjunto de entrenamiento usa 100 ejemplos sinteticos
- el conjunto de prueba usa 25 ejemplos sinteticos
- los datos se generan directamente en JavaScript

El repositorio tambien incluye CSV en `data/celsius_fahrenheit/`, pero la implementacion activa no los consume directamente. Funcionan como recurso de apoyo o referencia.

## Rutas disponibles

- `GET /`: pagina principal del simulador
- `GET /api/health`: responde `{"ok": true}`
- `GET /favicon.ico`: devuelve `assets/img/favicon.png` si existe

## Ejecucion local

### Requisitos

- Python 3.10 o superior recomendado
- conexion a internet para cargar TensorFlow.js desde CDN

### Instalacion

```powershell
pip install -r requirements.txt
```

### Arranque

```powershell
python Entendiendo_una_Red_Neuronal.py
```

Por defecto usa el puerto `5050`.

Si quieres cambiarlo:

```powershell
$env:PORT="8000"
python Entendiendo_una_Red_Neuronal.py
```

## Lo mas valioso del proyecto

Este repositorio tiene valor por dos razones complementarias:

- la demo activa ya sirve para explicar de forma muy visual como aprende una red sencilla
- la estructura del codigo y la documentacion revelan una base preparada para una plataforma educativa mas amplia

Es decir: no solo existe una app funcional, tambien existe una linea clara de crecimiento.

## Limitaciones actuales

Conviene documentar con honestidad el estado actual:

- La interfaz activa solo expone un caso practico visible.
- `learning_app.js` y `app/educational_content.py` todavia no estan integrados en la vista principal.
- `templates/login.html` es solo una plantilla minima.
- `app/api/` y `app/db/` son placeholders estructurales.
- TensorFlow.js depende de una carga externa por CDN.

## Evolucion natural posible

Por la base ya existente, el crecimiento mas coherente del proyecto seria:

- integrar varios ejemplos didacticos en la interfaz
- reutilizar `app/educational_content.py` como fuente real del contenido
- conectar o fusionar `learning_app.js` con la app actual
- crear rutas formativas adicionales en Flask
- aprovechar de forma explicita los datasets de `data/`

## Licencia

El proyecto esta publicado bajo licencia MIT.

## Resumen

`Entendiendo_una_Red_Neuronal` es una aplicacion educativa web orientada a explicar, con claridad visual, como funciona una red neuronal basica. En su estado actual ya es una demo interactiva util y bien enfocada. Al mismo tiempo, el repositorio contiene suficiente estructura, contenido y codigo adelantado como para evolucionar hacia una plataforma didactica bastante mas completa.
