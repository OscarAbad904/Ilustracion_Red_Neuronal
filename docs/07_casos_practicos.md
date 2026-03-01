# Casos practicos

## 1. Celsius -> Fahrenheit

- Tipo: regresion lineal simple.
- La red debe aprender una recta.
- Configuracion recomendada: 0 capas ocultas, salida lineal.
- Error tipico: usar sigmoid en la salida y recortar temperaturas altas.

## 2. Sensacion termica

- Tipo: regresion no lineal.
- La red debe mezclar temperatura, humedad y viento.
- Configuracion recomendada: 1 capa oculta de 8 neuronas, ReLU, salida lineal.
- Error tipico: usar 0 capas ocultas y forzar una relacion demasiado rigida.

## 3. Spam / no spam

- Tipo: clasificacion binaria.
- La red debe devolver una probabilidad de spam.
- Configuracion recomendada: 1 capa oculta de 6 neuronas, Tanh, salida sigmoid.
- Error tipico: salida lineal y lectura confusa de la probabilidad.

## 4. Flor simplificada

- Tipo: clasificacion multiclase.
- La red debe repartir probabilidad entre tres clases.
- Configuracion recomendada: 1 capa oculta de 10 neuronas, ReLU, salida Softmax.
- Error tipico: usar sigmoid por clase y perder una comparacion clara entre clases.

## 5. XOR

- Tipo: clasificacion binaria no lineal.
- La red debe activar solo si las entradas son distintas.
- Configuracion recomendada: 1 capa oculta de 2 neuronas, Tanh, salida sigmoid.
- Error tipico: intentar resolverlo con 0 capas ocultas.
