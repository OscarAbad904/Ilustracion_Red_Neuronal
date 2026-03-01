# Perdida y optimizacion

La perdida mide lo lejos que esta la prediccion del objetivo.

Sin perdida no hay criterio de mejora.

Casos tipicos:

- `MSE`: buena base para regresion. Penaliza mas los errores grandes.
- `MAE`: tambien sirve en regresion si quieres una lectura mas robusta a valores extremos.
- `Binary cross entropy`: natural para clasificacion binaria con salida sigmoid.
- `Categorical cross entropy`: natural para multiclase con Softmax.

Optimizacion, explicado sin formulas densas:

1. La red hace un forward.
2. Se calcula la perdida.
3. Se estima como cambia la perdida si ajustas cada peso.
4. Los pesos se mueven en la direccion que reduce el error.

Si la perdida no baja, no asumas que "faltan mas capas". A menudo el problema real es:

- salida mal elegida,
- activacion saturada,
- arquitectura innecesariamente profunda,
- o una tarea lineal tratada como si no lo fuera.
