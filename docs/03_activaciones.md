# Activaciones

La activacion es la parte que evita que toda la red sea solo una suma grande.

Resumen rapido:

- `Lineal`: salida sin curva. Buena para regresion.
- `Sigmoid`: comprime entre 0 y 1. Util para binaria.
- `Tanh`: comprime entre -1 y 1. Mejor centrada que sigmoid.
- `ReLU`: corta negativos. Rapida y muy usada.
- `Leaky ReLU`: deja una fuga en negativos para evitar neuronas muertas.
- `ELU`: suaviza negativos, mas estable que ReLU en algunos rangos.
- `Softplus`: version suave de ReLU.
- `GELU`: activacion moderna y suave.
- `Swish`: deja pasar informacion con una compuerta suave.

Reglas de salida:

- Regresion: usa salida lineal.
- Binaria: usa salida sigmoid.
- Multiclase: usa Softmax para repartir probabilidad total.

Fallo habitual:

Usar la misma activacion por costumbre en todas las capas y en todos los problemas. La activacion correcta depende del tipo de salida y del rango donde esperas operar.
