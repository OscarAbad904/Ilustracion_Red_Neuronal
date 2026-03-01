# Forward pass

El `forward pass` es el recorrido de la informacion desde la entrada hasta la salida.

Secuencia:

1. Entra un vector de datos.
2. Cada capa calcula sus sumas ponderadas.
3. Cada neurona aplica su activacion.
4. La ultima capa produce la prediccion.

Eso ya permite responder una pregunta: "Con los pesos actuales, que cree la red?"

Ejemplos:

- En `Celsius -> Fahrenheit`, el forward ideal reproduce una recta.
- En `spam / no spam`, el forward ideal produce una probabilidad entre 0 y 1.
- En `flor multiclase`, el forward ideal reparte probabilidad entre tres clases.

Error comun:

Confundir "la red calcula una salida" con "la red ya ha aprendido". El forward solo muestra la prediccion actual. Aprender implica comparar esa salida con el objetivo y corregir pesos.
