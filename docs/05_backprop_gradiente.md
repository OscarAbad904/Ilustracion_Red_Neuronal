# Backprop y gradiente

`Backpropagation` es el mecanismo que reparte la correccion desde la salida hacia atras.

Idea central:

- Si la salida falla, la red necesita saber que pesos han contribuido a ese fallo.
- El gradiente transporta esa senal de correccion capa por capa.

Lectura didactica del estado del gradiente:

- `Fuerte`: la correccion aun llega con margen util.
- `Debil`: la red puede aprender, pero cada ajuste sera pequeno.
- `Bloqueado`: casi no hay senal de correccion.

Por que se bloquea:

- Muchas `sigmoid` o `tanh` fuera de su zona central.
- Muchas `ReLU` recibiendo valores negativos y quedando en cero.
- Demasiada profundidad para un problema sencillo.

Caso importante:

XOR con 0 capas ocultas puede fallar aunque el gradiente no parezca el problema. Eso demuestra que no todo error viene de backprop: a veces la arquitectura simplemente no puede representar la solucion.
