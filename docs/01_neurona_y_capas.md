# Neurona y capas

Una neurona artificial recibe varias entradas, multiplica cada una por un peso, suma un bias y luego aplica una activacion.

Lectura practica:

- `entrada`: dato que llega desde el problema.
- `peso`: cuanto influye cada entrada.
- `bias`: desplazamiento fijo que ajusta el punto de respuesta.
- `activacion`: curva que convierte una suma en una respuesta util.

Una capa es un conjunto de neuronas que reciben la misma informacion de entrada, pero aprenden reglas distintas.

Por que varias capas:

- Con 0 capas ocultas, solo puedes aprender una transformacion directa.
- Con 1 capa oculta, ya puedes crear curvatura y separar patrones no lineales.
- Con muchas capas, ganas expresividad, pero tambien aumentas el riesgo de gradiente debil o bloqueado.

Regla didactica:

Empieza por la arquitectura mas pequena que pueda resolver el problema. Si la tarea es lineal, anadir profundidad suele empeorar la interpretacion antes que mejorar el resultado.
