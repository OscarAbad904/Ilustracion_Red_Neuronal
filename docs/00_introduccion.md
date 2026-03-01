# Introduccion

Esta aplicacion no intenta competir con librerias de entrenamiento. Su objetivo es mas concreto: ayudarte a entender que hace una red neuronal cuando recibe una entrada, la transforma capa a capa, produce una salida y corrige sus pesos.

La idea clave es separar tres preguntas:

1. Que problema intentas resolver.
2. Que salida tiene sentido para ese problema.
3. Que arquitectura minima basta para aprender el patron.

Por eso la app usa datos sinteticos. Asi se puede estudiar el mecanismo sin ruido adicional:

- Regresion lineal simple para ver el caso minimo.
- Regresion no lineal para justificar una capa oculta.
- Clasificacion binaria para introducir probabilidades.
- Clasificacion multiclase para mostrar Softmax.
- XOR para demostrar el limite de una sola capa.

Si una explicacion parece simple, es intencional. La meta no es esconder complejidad con jerga, sino exponer la causa real de cada decision.
