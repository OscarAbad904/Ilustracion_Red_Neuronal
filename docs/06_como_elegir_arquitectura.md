# Como elegir arquitectura

No elijas profundidad por intuicion visual. Elige por la forma del problema.

Guia corta:

1. Si la relacion es lineal, prueba sin capas ocultas.
2. Si hay curvatura o interacciones, empieza con 1 capa oculta.
3. Si el problema es binario, revisa primero la salida antes de aumentar neuronas.
4. Si el problema es multiclase, revisa si la salida debe comparar clases entre si.

Configuraciones base recomendadas:

- `Celsius -> Fahrenheit`: 0 capas ocultas, salida lineal.
- `Sensacion termica`: 1 capa oculta de 8 neuronas, ReLU o Swish, salida lineal.
- `Spam / no spam`: 1 capa oculta de 6 neuronas, Tanh, salida sigmoid.
- `Flor multiclase`: 1 capa oculta de 10 neuronas, ReLU, salida Softmax.
- `XOR`: 1 capa oculta de 2 neuronas, Tanh, salida sigmoid.

Regla practica:

Si el modelo falla, primero revisa:

- si la salida coincide con el problema,
- si la activacion deja pasar gradiente,
- y si la tarea realmente exige profundidad.

Solo despues tiene sentido probar mas capas o mas neuronas.
