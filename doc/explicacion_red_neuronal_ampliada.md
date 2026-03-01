# Explicacion ampliada y mas clara sobre redes neuronales

Este archivo recoge el contenido esencial del PDF `Explicacion didactica y rigurosa de que compone una red neuronal, como funciona y que elegir segun el caso`, pero reescrito en formato Markdown y ampliado para que una persona que esta empezando pueda entenderlo sin necesitar experiencia previa.

La idea es conservar el contenido del documento original, pero explicar mejor los conceptos que alli aparecen muy condensados o con terminologia tecnica.

---

## 1. Resumen ejecutivo explicado

Una red neuronal puede verse como una maquina que recibe datos de entrada, hace varios calculos intermedios y devuelve una salida.

En el PDF original se dice que una red neuronal moderna es "una funcion compuesta formada por muchas transformaciones parametrizadas por pesos". Esa definicion es correcta, pero para una persona principiante conviene traducirla asi:

- Una **funcion** es una regla que transforma una cosa en otra.
- Una **funcion compuesta** es una cadena de varias reglas seguidas.
- Los **pesos** son numeros ajustables que determinan cuanto influye cada entrada.

Ejemplo sencillo:

- Entrada: temperatura en Celsius = `25`
- Regla intermedia 1: multiplicar por un peso
- Regla intermedia 2: sumar un sesgo
- Regla final: producir una prediccion en Fahrenheit

La red hace esto en dos fases:

1. **Forward pass**: pasa la informacion hacia delante para calcular una prediccion.
2. **Entrenamiento**: compara la prediccion con el valor real y corrige los pesos para cometer menos error la siguiente vez.

Ese ciclo es el nucleo del aprendizaje:

1. entra un dato
2. la red predice
3. se calcula el error
4. se ajustan los pesos
5. se repite muchas veces

---

## 2. De que compone una red neuronal

## 2.1 Entradas

Las entradas son los datos que la red recibe.

Pueden ser:

- numeros sueltos
- una lista de valores
- pixeles de una imagen
- palabras convertidas en numeros
- una serie temporal

Ejemplo:

- Si quieres predecir Fahrenheit a partir de Celsius, la entrada puede ser solo un numero: `25`.
- Si quieres predecir el precio de una casa, las entradas pueden ser: metros cuadrados, numero de habitaciones y antiguedad.

## 2.2 Tensores

El PDF usa la palabra **tensor**. Es un termino correcto, pero muchas veces se introduce sin explicar.

Una forma simple de entenderlo:

- un escalar es un numero: `5`
- un vector es una lista de numeros: `[2, 7, 9]`
- una matriz es una tabla de numeros
- un tensor es el nombre general para cualquiera de esas estructuras cuando trabajamos con redes neuronales

No hay magia aqui. Muchas veces "tensor" solo significa "un bloque de numeros".

## 2.3 Pesos y sesgos

Los **pesos** indican cuanta importancia tiene cada entrada.

Los **sesgos** (bias) permiten desplazar el resultado aunque la entrada sea cero.

Formula basica de una neurona:

`z = w1*x1 + w2*x2 + ... + b`

Donde:

- `x` son las entradas
- `w` son los pesos
- `b` es el sesgo
- `z` es el resultado antes de aplicar la activacion

Ejemplo:

- `x = 10`
- `w = 1.8`
- `b = 32`

Entonces:

`z = 1.8 * 10 + 32 = 50`

En este ejemplo, la neurona ya esta haciendo exactamente la conversion de Celsius a Fahrenheit.

---

## 3. Representacion de entrada

## 3.1 Normalizacion

El PDF habla de **normalizacion de entrada** o **feature scaling**.

Esto significa cambiar la escala de los datos para que todos esten en rangos parecidos.

Por que ayuda:

- evita que una variable enorme domine a otra pequena
- hace que el entrenamiento sea mas estable
- suele acelerar la convergencia

Ejemplo:

- edad: `35`
- salario: `25000`

Si metes ambos valores tal cual, el salario puede "pesar" demasiado por ser numericamente mucho mayor.

Una normalizacion simple seria:

- edad normalizada = `35 / 100 = 0.35`
- salario normalizado = `25000 / 100000 = 0.25`

Ahora ambas variables se mueven en escalas parecidas.

## 3.2 Embeddings

El PDF menciona **embeddings**. Este concepto suele sonar abstracto al principio.

Explicacion simple:

- si una red recibe una categoria como "rojo", "verde" o "azul", no puede operar directamente con texto
- un embedding convierte cada categoria en una lista corta de numeros que la red puede aprender a usar

Una forma muy practica de entenderlo es compararlo con una ficha de identificacion compacta.

En lugar de guardar una categoria con una lista larguisima de ceros y un uno (como ocurre con one-hot), el embedding guarda unos pocos numeros que resumen como se comporta esa categoria para la tarea.

Ejemplo:

- "gato" -> `[0.2, -0.4, 0.8]`
- "perro" -> `[0.1, -0.3, 0.75]`
- "coche" -> `[-0.6, 0.9, -0.1]`

Si dos palabras acaban con vectores parecidos, significa que para la red tienen algun significado parecido en la tarea.

Ejemplo sencillo de tienda online:

- categoria: `camiseta`
- categoria: `sudadera`
- categoria: `destornillador`

Lo esperable es que la red termine aprendiendo embeddings mas parecidos entre `camiseta` y `sudadera` que entre `camiseta` y `destornillador`, porque en muchas tareas las dos primeras se parecen mas entre si.

La idea clave para una persona principiante es esta:

- un embedding no es mas que una forma compacta y aprendible de representar categorias con numeros utiles

## 3.3 Codificacion posicional

En el PDF aparece **codificacion posicional** al hablar de Transformers.

La idea es sencilla:

- si la red solo ve palabras convertidas en vectores, puede perder el orden
- no es lo mismo "Juan come pan" que "Pan come Juan"

La codificacion posicional anade una senal que dice en que posicion esta cada elemento de la secuencia.

---

## 4. Capas y bloques principales

## 4.1 Capa densa o MLP

Una capa densa conecta cada neurona con todas las neuronas de la capa anterior.

Es la forma mas clasica de red neuronal.

Ventajas:

- sencilla de entender
- util para datos tabulares
- buena como punto de partida

Limitacion:

- no aprovecha estructuras especiales como la forma de una imagen o el orden temporal de una secuencia

## 4.2 Capas convolucionales (CNN)

El PDF explica que las CNN usan filtros locales y comparten pesos.

Traduccion practica:

- en vez de mirar toda la imagen de golpe, la red mira pequenas ventanas
- usa el mismo detector en muchas zonas

Ejemplo:

- un filtro puede aprender a detectar bordes
- ese mismo detector de bordes se aplica en toda la imagen

Esto reduce parametros y aprovecha que en imagenes los patrones locales importan mucho.

## 4.3 Pooling

El **pooling** reduce el tamano de la informacion.

Ejemplo con max pooling:

- si en una zona tienes `[2, 8, 3, 1]`
- el max pooling se queda con `8`

Esto simplifica la representacion y reduce coste de calculo.

## 4.4 RNN, LSTM y GRU

Estas capas se usan para secuencias.

La idea importante:

- procesan un elemento tras otro
- mantienen una especie de "memoria" interna

Ejemplo:

- al leer una frase, lo que aparece antes influye en como interpretar lo que viene despues

La diferencia principal:

- **RNN**: la version basica
- **LSTM**: mejora la memoria en secuencias largas
- **GRU**: similar a LSTM, pero algo mas simple

## 4.5 Atencion

La palabra **atencion** suena compleja, pero la idea es muy intuitiva:

- cuando la red va a decidir algo, no trata todas las partes de la entrada como igual de importantes
- asigna mas importancia a unas partes que a otras

Ejemplo:

En la frase "El perro que persiguio al gato ladro", si queremos interpretar "ladro", la palabra "perro" pesa mas que "gato".

Eso es atencion: decidir en que partes conviene fijarse mas.

Otro ejemplo muy claro:

Si alguien pregunta: "Cual es la capital de Francia?", una persona no presta la misma atencion a todas las palabras. Se fija sobre todo en `capital` y `Francia`.

La red hace algo parecido:

- mira todos los elementos
- calcula cuales son mas importantes para esa respuesta concreta
- da mas peso a esos elementos

## 4.6 Transformers

Un Transformer es una arquitectura que usa atencion como pieza central.

Se hizo muy popular porque:

- maneja bien relaciones largas
- puede entrenarse en paralelo mejor que una RNN
- funciona muy bien en lenguaje y tambien en otras tareas

Pensarlo asi ayuda:

- una RNN lee paso a paso
- un Transformer puede comparar muchos elementos entre si al mismo tiempo

Una analogia util:

- una RNN se parece a una persona que lee una frase palabra por palabra y va recordando lo anterior
- un Transformer se parece mas a una persona que puede mirar toda la frase de una vez y relacionar directamente unas palabras con otras

Eso hace que sea especialmente bueno cuando una palabra depende de otra que esta muy lejos dentro de la misma frase.

## 4.7 Conexiones residuales

El PDF habla de **skip connections** o **residuales**.

Explicacion simple:

- en vez de obligar a cada capa a rehacer todo desde cero
- se le permite recibir tambien una copia de informacion anterior

Es como decir:

"ademas de aprender algo nuevo, conserva parte de lo que ya funcionaba"

Esto ayuda a entrenar redes profundas porque la informacion y los gradientes circulan mejor.

## 4.8 BatchNorm y LayerNorm

Ambas son tecnicas de normalizacion interna.

Su objetivo es que las activaciones de la red no se descontrolen demasiado mientras se entrena.

Diferencia sencilla:

- **BatchNorm**: usa estadisticas del lote de datos
- **LayerNorm**: normaliza dentro de cada muestra

En una primera aproximacion:

- BatchNorm es muy comun en vision
- LayerNorm es muy comun en Transformers

## 4.9 Dropout

El **dropout** apaga neuronas al azar durante el entrenamiento.

Aunque suene raro, esto obliga a la red a no depender demasiado de una sola ruta.

Analogamente:

- si un equipo solo funciona porque una persona lo hace todo, es fragil
- si varias personas pueden cubrir el trabajo, es mas robusto

Eso mismo busca el dropout.

## 4.10 Capas de salida

La ultima capa depende del tipo de problema.

- **Lineal**: para regresion
- **Sigmoid**: para si/no o varias etiquetas independientes
- **Softmax**: para elegir una clase entre varias opciones excluyentes

Ejemplos:

- predecir temperatura: salida lineal
- detectar si un correo es spam o no: sigmoid
- clasificar una imagen como gato, perro o pajaro: softmax

---

## 5. Como funciona el aprendizaje

## 5.1 Forward pass

En el **forward pass**, la informacion entra por la izquierda, cruza las capas y sale una prediccion.

Ejemplo muy simple con una neurona:

- entrada `x = 25`
- peso `w = 1.5`
- sesgo `b = 10`

Calculo:

- `z = 1.5 * 25 + 10 = 47.5`

Si la salida es lineal, la prediccion es `47.5`.

Si el valor real era `77`, la red todavia se equivoca.

## 5.2 Funcion de perdida

La **funcion de perdida** mide cuanto se equivoca la red.

No es la salida, sino el numero que resume el error.

Ejemplo:

- valor real: `77`
- prediccion: `47.5`

Error simple:

- diferencia = `77 - 47.5 = 29.5`

Una perdida habitual en regresion es MSE:

- error cuadratico = `(77 - 47.5)^2`

Cuanto mayor es la perdida, peor esta prediciendo la red.

## 5.3 Backpropagation

Este es uno de los puntos mas tecnicos del PDF y uno de los que mas conviene aclarar.

El PDF dice correctamente que backpropagation usa la regla de la cadena para calcular gradientes.

Explicacion simple:

- la red se equivoco
- necesitamos saber que pesos son responsables de ese error
- backpropagation calcula cuanto ha contribuido cada peso al error final
- con eso, cada peso se corrige un poco

No "inventa reglas". Solo calcula derivadas para saber en que direccion conviene mover cada parametro.

Analogamente:

- si una receta sale demasiado salada
- revisas cuanto aporto cada ingrediente
- luego ajustas la cantidad para la siguiente vez

Eso hace backpropagation con numeros.

Ejemplo numerico muy simple:

- la red predice `60`
- el valor real es `77`
- hay error
- uno de los pesos vale `1.5`

Si el calculo del gradiente dice que ese peso esta haciendo que la salida se quede demasiado baja, la correccion puede ser subirlo un poco, por ejemplo de `1.5` a `1.6`.

La idea importante no es memorizar la formula completa, sino entender esto:

- backpropagation le dice a cada peso si debe subir, bajar o quedarse casi igual

## 5.4 Gradiente

El **gradiente** indica hacia donde cambia el error.

Interpretacion practica:

- si un gradiente es positivo, quiza haya que bajar ese peso
- si es negativo, quiza haya que subirlo
- si es casi cero, mover ese peso apenas cambia el error

No hace falta pensar en ello como un concepto abstracto de calculo vectorial al principio. Basta verlo como "la pista que dice como corregir los parametros".

Una analogia util es imaginar una ladera:

- si estas en una montana y quieres bajar, miras hacia donde esta la pendiente
- el gradiente te dice precisamente eso: hacia donde cambia mas el terreno

En redes neuronales, en lugar de bajar una montana, queremos bajar el error.

## 5.5 Optimizador

El optimizador es la regla concreta que actualiza los pesos usando los gradientes.

El PDF cita varios:

- SGD
- Momentum
- RMSProp
- Adam
- AdamW

Explicacion sencilla:

- **SGD**: da pequenos pasos segun el gradiente
- **Momentum**: acumula inercia para no zigzaguear tanto
- **RMSProp**: ajusta el tamano del paso segun el historial reciente
- **Adam**: combina inercia y ajuste adaptativo del paso
- **AdamW**: version de Adam con una regularizacion mejor separada

Regla practica para principiante:

- si quieres algo facil para empezar: Adam
- si buscas una base clasica y robusta: SGD con momentum

---

## 6. Funciones de activacion

Sin una funcion de activacion no lineal, varias capas seguidas equivaldrian a una sola cuenta lineal.

Eso significa que la red perderia gran parte de su capacidad para aprender relaciones complejas.

## 6.1 Sigmoid

Convierte el valor en un numero entre `0` y `1`.

Se usa mucho cuando queremos interpretar la salida como probabilidad.

Problema habitual:

- en valores muy altos o muy bajos se "satura"
- eso hace que aprenda mas despacio en redes profundas

## 6.2 Tanh

Se parece a sigmoid, pero devuelve valores entre `-1` y `1`.

Ventaja:

- esta centrada en cero

Problema:

- tambien puede saturarse

## 6.3 ReLU

Es una de las mas usadas.

Regla:

- si el valor es negativo, devuelve `0`
- si es positivo, lo deja pasar

Ventajas:

- simple
- rapida
- suele entrenar bien

Problema:

- algunas neuronas pueden quedarse siempre en `0`

## 6.4 Leaky ReLU, PReLU, GELU y Swish

Son variaciones para mejorar algunos problemas de ReLU o para lograr mejor rendimiento.

Idea corta:

- **Leaky ReLU**: deja pasar un poco de senal negativa
- **PReLU**: ese "poco" se aprende
- **GELU** y **Swish**: son mas suaves y se usan mucho en arquitecturas modernas

---

## 7. Activacion de salida y perdida: que combinar

Este punto es muy importante en la practica.

La salida y la perdida deben "encajar" entre si.

Una forma sencilla de entenderlo:

- la salida define en que formato habla la red
- la perdida define como medimos si ese formato esta bien o mal

Si ambas no encajan, el entrenamiento se vuelve confuso o poco estable.

## 7.1 Clasificacion binaria

- salida: `sigmoid`
- perdida: `binary cross-entropy`

Ejemplo:

- "spam" o "no spam"

Aqui tiene sentido porque `sigmoid` da un valor entre `0` y `1`, que podemos leer como probabilidad.

## 7.2 Clasificacion multiclase excluyente

- salida: `softmax`
- perdida: `cross-entropy`

Ejemplo:

- clasificar una imagen como `gato`, `perro` o `caballo`

Aqui `softmax` reparte la probabilidad entre varias clases y obliga a que todas sumen `1`.

## 7.3 Regresion

- salida: `lineal`
- perdida: `MSE` o `MAE`

Ejemplo:

- predecir temperatura
- predecir precio

Diferencia simple:

- **MSE** castiga mas los errores grandes
- **MAE** es mas robusta si hay valores atipicos

Ejemplo claro:

Si quieres predecir grados Fahrenheit, no tiene sentido usar `softmax`, porque no quieres elegir una clase entre varias, sino devolver un numero real como `77.3`.

Por eso, para este tipo de problema, lo normal es:

- salida lineal
- una perdida de regresion

## 7.4 Segmentacion y deteccion

El PDF tambien cubre tareas mas avanzadas.

- **segmentacion**: predecir una clase por pixel
- **deteccion**: encontrar objetos y sus cajas

Aqui ya se combinan varias salidas y varias perdidas a la vez.

Para una persona que empieza, lo importante es quedarse con esta idea:

- cuanto mas compleja es la tarea, mas normal es que la red tenga varias partes que aprenden cosas distintas al mismo tiempo

---

## 8. Inicializacion, regularizacion y sobreajuste

## 8.1 Inicializacion de pesos

No conviene empezar con pesos cualquiera.

Si los pesos iniciales son demasiado grandes o demasiado pequenos:

- las activaciones se disparan
- o se vuelven insignificantes
- y entrenar se vuelve dificil

Por eso existen inicializaciones conocidas:

- **Xavier/Glorot**: suele ir bien con activaciones como tanh
- **He**: suele ir bien con ReLU

No hace falta memorizar todas las formulas al principio. Lo importante es entender que la inicializacion influye mucho en si la red aprende o no.

## 8.2 Overfitting y underfitting

Estas dos palabras aparecen mucho y suelen explicarse demasiado rapido.

**Overfitting**:

- la red aprende muy bien el entrenamiento
- pero falla en datos nuevos

Es como memorizar respuestas exactas sin haber entendido el tema.

Ejemplo:

- entrenas con 100 fotos concretas de gatos y perros
- la red memoriza detalles muy especificos de esas fotos
- cuando le enseñas fotos nuevas, se equivoca mas de lo esperado

**Underfitting**:

- la red ni siquiera aprende bien el entrenamiento

Es como intentar resolver un problema dificil con una herramienta demasiado simple o mal configurada.

Ejemplo:

- quieres aprender una relacion compleja
- pero usas una red demasiado pequena o entrenas muy poco
- el modelo falla incluso con los datos que ya habia visto

Regla rapida:

- si falla solo fuera del entrenamiento, sospecha de overfitting
- si falla tanto dentro como fuera, sospecha de underfitting

## 8.3 Regularizacion

La regularizacion son tecnicas para evitar que la red se ajuste "demasiado" al entrenamiento.

Herramientas tipicas:

- dropout
- weight decay
- early stopping
- data augmentation

## 8.4 Weight decay

El PDF menciona que penaliza pesos grandes.

Explicacion simple:

- si la red necesita pesos exageradamente grandes para funcionar, muchas veces esta forzando demasiado el ajuste
- penalizar pesos grandes ayuda a que aprenda soluciones mas estables

## 8.5 Early stopping

Significa parar el entrenamiento cuando la validacion deja de mejorar.

Idea practica:

- si el entrenamiento sigue mejorando pero la validacion ya no
- seguir entrenando puede empeorar la generalizacion

---

## 9. Como detectar si la red va bien o mal

Una forma muy util es mirar dos curvas:

- perdida en entrenamiento
- perdida en validacion

Interpretacion rapida:

- si entrenamiento baja y validacion tambien baja: va bien
- si entrenamiento baja pero validacion sube: hay overfitting
- si ambas estan mal: hay underfitting o el entrenamiento esta mal configurado

Esta es una de las herramientas mas practicas para tomar decisiones.

---

## 10. Metricas segun el tipo de problema

## 10.1 Clasificacion

- **Accuracy**: porcentaje de aciertos
- **Precision**: de lo que marco como positivo, cuanto era realmente positivo
- **Recall**: de lo que era realmente positivo, cuanto detecte
- **F1**: equilibrio entre precision y recall
- **ROC-AUC**: mide capacidad de separar clases
- **PR-AUC**: muy util cuando hay mucho desbalanceo

Ejemplo:

Si detectas enfermedad rara, accuracy puede enganar.

Si 99 personas estan sanas y 1 esta enferma, una red que diga "todos sanos" tiene 99% de accuracy, pero no sirve.

Ejemplo aun mas concreto:

- el modelo marca 10 casos como positivos
- de esos 10, solo 6 lo eran de verdad -> precision = 6/10
- en realidad habia 8 positivos reales
- si detecto 6 de esos 8 -> recall = 6/8

Esto suele aclarar mejor la diferencia entre precision y recall.

## 10.2 Regresion

- **MSE / RMSE**: castigan mas los errores grandes
- **MAE**: mide error absoluto medio
- **R2**: indica cuanta variacion explica el modelo

Ejemplo:

Si el valor real es `77` y predices `76`, el error es pequeno.

Si predices `20`, el error es muy grande y MSE lo castigara mucho mas.

Explicacion corta de `R2`:

- si `R2` esta cerca de `1`, el modelo explica bastante bien la variacion de los datos
- si `R2` esta cerca de `0`, apenas mejora frente a una prediccion muy basica
- si es negativo, el modelo lo esta haciendo peor de lo esperado

## 10.3 Segmentacion

- **IoU**: interseccion sobre union
- **Dice**: otra medida de solapamiento

Ambas comparan cuanto coincide la zona predicha con la zona real.

Ejemplo simple:

- si el modelo pinta casi la misma zona que la mascara correcta, IoU y Dice suben
- si pinta una zona muy distinta, bajan

## 10.4 Deteccion

- **mAP**: una medida global muy usada en deteccion de objetos

No hace falta dominarla al principio. Basta saber que resume la calidad del detector considerando precision y recall.

Una forma simple de pensarlo es esta:

- valora si el detector encuentra los objetos correctos
- y tambien si los encierra razonablemente bien con sus cajas

---

## 11. Que arquitectura elegir segun el caso

El PDF incluye una guia rapida muy util. La reformulo de forma mas directa:

## 11.1 Datos tabulares

Usa primero:

- una MLP pequena o mediana
- buen preprocesado
- normalizacion

Si hay categorias grandes:

- considera embeddings

## 11.2 Imagen

Usa primero:

- CNN con residuales

Si hay poco dato:

- suele ayudar el transfer learning

## 11.3 Segmentacion

Usa:

- U-Net o FCN

Porque deben predecir pixel a pixel y conservar detalle.

## 11.4 Deteccion de objetos

Usa como referencia conceptual:

- Mask R-CNN

## 11.5 Texto

Usa:

- Transformer
- fine-tuning de un modelo preentrenado tipo BERT

## 11.6 Series temporales

Opciones comunes:

- LSTM o GRU si importa la memoria paso a paso
- Transformer si necesitas manejar dependencias largas y entrenamiento mas paralelo
- 1D-CNN si dominan patrones locales

---

## 12. Transfer learning explicado sin tecnicismos

El **transfer learning** consiste en reutilizar un modelo ya entrenado en una tarea parecida.

Ejemplo:

- en vez de entrenar desde cero un modelo de imagen
- empiezas con uno que ya aprendio rasgos generales como bordes, texturas y formas
- luego lo ajustas a tu problema concreto

Ejemplo muy claro:

- un modelo fue entrenado para distinguir miles de objetos comunes
- tu ahora quieres detectar defectos en piezas metalicas
- aunque no sean exactamente las mismas imagenes, ese modelo ya sabe reconocer lineas, contornos y texturas
- eso te da una base mucho mejor que empezar desde cero

Esto suele ser buena idea cuando:

- tienes pocos datos
- entrenar desde cero es caro
- el problema nuevo se parece razonablemente al original

La idea intuitiva es:

- primero aprovechas conocimiento general
- despues haces un ajuste fino a tu caso

Es parecido a contratar a una persona que ya sabe conducir y solo necesita aprender una ruta nueva, en lugar de ensenar a conducir desde cero.

---

## 13. Pseudocodigo explicado

El PDF incluye pseudocodigo de forward, backprop y Adam. Aqui va la misma idea con explicacion adicional.

## 13.1 Forward

```text
h1 = W1 * x + b1
a1 = ReLU(h1)
h2 = W2 * a1 + b2
y_hat = salida(h2)
L = perdida(y_real, y_hat)
```

Lectura sencilla:

1. la primera capa mezcla las entradas
2. la activacion introduce no linealidad
3. la segunda capa produce una prediccion
4. se compara con el valor real

## 13.2 Backprop

```text
delta_salida = derivada_de_la_perdida
grad_W2 = delta_salida * a1
delta_1 = (W2^T * delta_salida) * derivada_ReLU(h1)
grad_W1 = delta_1 * x
```

Lectura sencilla:

1. se empieza por el error de salida
2. se calcula como afecta a la ultima capa
3. ese error se reparte hacia atras
4. se calcula como afecta a capas anteriores

## 13.3 Actualizacion con Adam

Adam no solo mira el gradiente actual.

Tambien recuerda:

- una media del gradiente reciente
- una media de la magnitud reciente del gradiente

Eso le permite ajustar mejor el tamano del paso.

La intuicion es mas importante que la formula completa al empezar:

- Adam intenta moverse rapido cuando puede
- y ser mas prudente cuando detecta cambios inestables

---

## 14. Glosario de terminos tecnicos del PDF

Aqui estan varios terminos del PDF explicados en lenguaje simple:

- **Parametro**: numero ajustable de la red, como un peso o un sesgo.
- **Feature**: una caracteristica de entrada, como edad o salario.
- **Backbone**: la parte principal del modelo que extrae patrones.
- **Head**: la parte final que convierte lo aprendido en la salida concreta.
- **Logit**: valor previo a aplicar sigmoid o softmax.
- **Gradiente**: senal que indica como cambiar un parametro para bajar el error.
- **Minibatch**: pequeno grupo de ejemplos usado en una actualizacion.
- **Epoch**: una pasada completa por todo el conjunto de entrenamiento.
- **Generalizacion**: capacidad de funcionar bien con datos nuevos.
- **Inferencia**: usar la red ya entrenada para predecir.
- **Hiperparametro**: ajuste elegido por la persona, como learning rate o numero de capas.
- **Capacidad del modelo**: lo complejo que puede llegar a ser lo que aprende.
- **Saturacion**: zona donde una activacion cambia muy poco y el aprendizaje se frena.
- **Neurona muerta**: neurona que deja de activarse y apenas contribuye.
- **Schedule**: plan para cambiar el learning rate durante el entrenamiento.

---

## 15. Regla practica para principiantes

Si una persona esta empezando, esta secuencia de decision suele ser suficiente:

1. define bien la tarea: clasificacion, regresion, segmentacion o deteccion
2. elige una arquitectura base simple para esa tarea
3. normaliza los datos
4. usa una salida y una perdida compatibles
5. empieza con Adam o SGD con momentum
6. vigila entrenamiento y validacion
7. si aparece overfitting, regulariza
8. si no aprende, revisa datos, escala, learning rate y arquitectura

La mayor parte de los problemas reales no se arreglan con una formula "magica", sino entendiendo bien estos pasos basicos.

---

## 16. Fuentes citadas en el PDF original

El PDF original termina con una lista de referencias. A continuacion se dejan recogidas como parte del contenido base:

- Goodfellow et al.: Deep Learning Book Chapter 6
- How transferable are features in deep neural networks?
- Understanding the difficulty of training deep feedforward neural networks
- Efficient Estimation of Word Representations in Vector Space
- Attention Is All You Need
- Gradient-Based Learning Applied to Document Recognition
- Long Short-Term Memory
- Neural Machine Translation by Jointly Learning to Align and Translate
- Deep Residual Learning for Image Recognition
- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- Layer Normalization
- Dropout: A Simple Way to Prevent Neural Networks from Overfitting
- Learning representations by back-propagating errors
- A Stochastic Approximation Method
- On the importance of initialization and momentum in deep learning
- Adam: A Method for Stochastic Optimization
- The Marginal Value of Adaptive Gradient Methods in Machine Learning
- Rectified Linear Units Improve Restricted Boltzmann Machines
- Delving Deep into Rectifiers
- Gaussian Error Linear Units (GELUs)
- Searching for Activation Functions
- mean_squared_error - scikit-learn
- mean_absolute_error - scikit-learn
- r2_score - scikit-learn
- precision_recall_fscore_support - scikit-learn
- roc_auc_score - scikit-learn
- Precision-Recall - scikit-learn
- Fully Convolutional Networks for Semantic Segmentation
- U-Net: Convolutional Networks for Biomedical Image Segmentation
- Measures of the Amount of Ecologic Association Between Species
- Microsoft COCO: Common Objects in Context
- The Pascal Visual Object Classes (VOC) Challenge
- Mask R-CNN
- Decoupled Weight Decay Regularization
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- Early Stopping | but when?

---

## 17. Cierre

La version original del PDF es correcta y rigurosa, pero en varios puntos asume que el lector ya conoce terminos como gradiente, embeddings, atencion, residual, logits o generalizacion.

Esta version en Markdown mantiene el contenido principal, pero lo aterriza con:

- lenguaje mas directo
- pasos secuenciales
- analogias sencillas
- ejemplos numericos simples

Esta guia ya incluye una base mas clara que el PDF original. A partir de aqui, se han anadido tambien una tabla muy simple de optimizadores, un ejemplo completo resuelto y una version adicional todavia mas basica para formacion inicial.


---

## 18. Tabla muy simple de optimizadores para principiantes

Esta tabla no pretende ser matematica ni exhaustiva. Solo sirve para entender, de manera rapida, cuando usar cada opcion sin perderse en tecnicismos.

| Optimizador | Idea sencilla | Ventaja principal | Riesgo o limite | Cuando suele ser buena primera opcion |
| --- | --- | --- | --- | --- |
| SGD | Mueve los pesos poco a poco siguiendo el gradiente | Es simple y muy clasico | Puede ir lento si no ajustas bien el learning rate | Cuando quieres una base sencilla y controlada |
| SGD + momentum | Igual que SGD, pero con "inercia" para no zigzaguear tanto | Suele ser mas estable y rapido que SGD solo | Requiere algo mas de ajuste | Muy buena opcion clasica en muchos problemas |
| RMSProp | Ajusta el tamano del paso segun el historial reciente | Puede arrancar bien cuando el entrenamiento es irregular | No siempre generaliza igual de bien | Como prueba intermedia si SGD va tosco |
| Adam | Combina inercia y ajuste adaptativo del paso | Suele ser el mas facil para empezar | Puede converger rapido pero no siempre dar la mejor generalizacion final | Primera opcion muy practica para principiantes |
| AdamW | Similar a Adam, pero con mejor control del weight decay | Suele encajar muy bien en modelos modernos | Sigue necesitando ajuste | Muy habitual en practicas actuales y Transformers |

Regla practica muy simple:

- si no sabes por donde empezar, prueba `Adam`
- si quieres una base clasica y robusta, prueba `SGD + momentum`
- si usas modelos modernos grandes, `AdamW` suele ser una eleccion razonable

---

## 19. Ejemplo completo resuelto: red 1 entrada -> 1 capa oculta -> 1 salida

Aqui va un ejemplo totalmente didactico, con numeros pequenos y pasos visibles.

Objetivo del ejemplo:

- entrada: temperatura en Celsius
- salida: temperatura estimada en Fahrenheit

Para que los numeros sean faciles de manejar, vamos a trabajar con la entrada normalizada.

### 19.1 Estructura de la red

Usaremos esta red sencilla:

- 1 entrada: `x`
- 1 neurona en la capa oculta
- 1 neurona de salida

Esquema:

`x -> h1 -> y`

Elegimos estos valores iniciales:

- entrada real: `25 C`
- entrada normalizada: `x = 0.25`
- peso de entrada a capa oculta: `w1 = 2.0`
- sesgo de capa oculta: `b1 = 0.10`
- activacion oculta: `ReLU`
- peso de capa oculta a salida: `w2 = 40`
- sesgo de salida: `b2 = 30`
- salida final: lineal

Valor real esperado:

- `F = 25 * 9 / 5 + 32 = 77`

### 19.2 Paso 1: calculo en la capa oculta

Primero calculamos el valor antes de la activacion:

`z1 = w1 * x + b1`

Sustituyendo:

`z1 = 2.0 * 0.25 + 0.10 = 0.50 + 0.10 = 0.60`

Ahora aplicamos ReLU:

`a1 = ReLU(0.60) = 0.60`

Como el valor es positivo, ReLU no lo cambia.

### 19.3 Paso 2: calculo en la salida

Ahora usamos la salida de la capa oculta para calcular la prediccion final:

`y = w2 * a1 + b2`

Sustituyendo:

`y = 40 * 0.60 + 30 = 24 + 30 = 54`

La red predice:

- `54 F`

### 19.4 Paso 3: comparar con el valor real

Sabemos que el valor correcto era:

- `77 F`

Entonces la red esta fallando por:

- error simple = `77 - 54 = 23`

Eso significa que la prediccion se ha quedado corta.

### 19.5 Paso 4: calcular la perdida

Si usamos una perdida cuadratica simple:

`Loss = (77 - 54)^2`

Entonces:

`Loss = 23^2 = 529`

Ese numero no es la prediccion, sino una medida de cuanto se ha equivocado la red.

### 19.6 Paso 5: interpretar la correccion

Como la salida se ha quedado demasiado baja, necesitamos que la proxima vez la red produzca un valor mayor.

Eso puede lograrse, por ejemplo, de estas formas:

- subir `w2`
- subir `b2`
- hacer que `a1` salga mas grande, ajustando `w1` o `b1`

Supongamos que, tras backpropagation, el sistema decide hacer una pequena correccion:

- `w2` pasa de `40` a `42`

### 19.7 Paso 6: volver a calcular con el peso ajustado

Repetimos solo la salida final:

`y_nuevo = 42 * 0.60 + 30 = 25.2 + 30 = 55.2`

Ahora la nueva prediccion es:

- `55.2 F`

Sigue estando lejos de `77`, pero ya es un poco mejor que `54`.

La idea del entrenamiento es justo esa:

- no arreglarlo todo de golpe
- ir corrigiendo poco a poco en la direccion correcta

### 19.8 Que demuestra este ejemplo

Este ejemplo deja ver, sin formulas complicadas, las cuatro ideas principales:

1. la red transforma la entrada en pasos
2. cada capa hace una cuenta
3. la salida se compara con el valor real
4. los pesos se ajustan un poco para reducir el error

### 19.9 Que pasaria en una red real

En una red real:

- habria muchas neuronas, no una sola
- habria muchos pesos que ajustar a la vez
- el proceso se repetiria con muchos ejemplos
- los ajustes los haria automaticamente el algoritmo de entrenamiento

Pero el principio sigue siendo exactamente el mismo que en este ejemplo pequeno.

---

## 20. Como leer una red neuronal sin perderse

Cuando una persona empieza, a veces se bloquea porque ve demasiados simbolos a la vez. Una forma de leer cualquier red sin agobiarse es seguir siempre este orden:

1. que entra
2. que cuenta hace cada capa
3. que sale
4. cuanto se equivoca
5. como corrige los pesos

Si sigues siempre ese orden, casi cualquier explicacion tecnica se vuelve mucho mas manejable.

Ejemplo de lectura rapida:

- entra `25 C`
- la capa oculta transforma ese valor
- la salida predice una temperatura
- se compara con `77 F`
- los pesos se retocan para acercarse a `77`

---

## 21. Documento complementario: version muy basica

Ademas de esta guia ampliada, se ha creado una version todavia mas simple pensada para una primera toma de contacto:

- `doc/explicacion_red_neuronal_muy_basica.md`

Esa version reduce aun mas los tecnicismos y prioriza una explicacion corta, directa y orientada a formacion inicial.
