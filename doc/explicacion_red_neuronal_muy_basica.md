# Red neuronal explicada de forma muy basica

Este documento esta pensado para alguien que empieza desde cero.

La idea no es dar una definicion tecnica, sino entender que hace una red neuronal, como aprende y que decisiones basicas hay que tomar.

---

## 1. Que es una red neuronal

Una red neuronal es un sistema que:

1. recibe datos
2. hace calculos con esos datos
3. produce una respuesta
4. se corrige si se equivoca

Ejemplo:

- entrada: `25` grados Celsius
- salida esperada: `77` grados Fahrenheit

La red intenta aprender a transformar el `25` en `77`.

---

## 2. Como funciona por dentro

Una red neuronal trabaja con neuronas artificiales.

Cada neurona hace una cuenta parecida a esta:

`resultado = entrada * peso + sesgo`

### 2.1 Que es un peso

El peso dice cuanta importancia tiene una entrada.

- si el peso es grande, esa entrada influye mucho
- si el peso es pequeno, influye poco

### 2.2 Que es un sesgo

El sesgo es un ajuste extra.

Sirve para mover el resultado aunque la entrada no cambie.

---

## 3. Como aprende

Aprender significa ajustar pesos y sesgos para equivocarse menos.

La red sigue este ciclo:

1. recibe una entrada
2. hace una prediccion
3. compara esa prediccion con el valor real
4. calcula el error
5. ajusta sus pesos
6. vuelve a intentarlo

Eso se repite muchas veces.

---

## 4. Ejemplo muy simple

Queremos convertir Celsius a Fahrenheit.

Sabemos que:

`F = C * 9 / 5 + 32`

Si entra:

- `C = 25`

El resultado correcto es:

- `F = 77`

Supongamos que la red al principio predice:

- `F = 60`

Entonces:

- se ha equivocado
- necesita corregirse

Si tras corregir un poco sus pesos en el siguiente intento predice `65`, ha mejorado.

Si luego predice `72`, ha mejorado otra vez.

El aprendizaje consiste en repetir esas pequenas correcciones hasta acercarse al resultado correcto.

---

## 5. Que es el error o perdida

La perdida es un numero que nos dice cuanto se ha equivocado la red.

- si la perdida es alta, va mal
- si la perdida baja, esta aprendiendo

No es la respuesta final.

Es solo la medida del fallo.

---

## 6. Que es backpropagation

Aunque el nombre suene complicado, la idea es simple.

Backpropagation sirve para decidir como corregir los pesos.

En otras palabras:

- si la red se quedo corta, algunos pesos tendran que subir
- si la red se paso, algunos pesos tendran que bajar

Eso es todo lo importante para empezar a entenderlo.

---

## 7. Que tipos de problemas puede resolver

### 7.1 Clasificacion

Elegir una categoria.

Ejemplos:

- spam o no spam
- gato o perro

### 7.2 Regresion

Predecir un numero.

Ejemplos:

- temperatura
- precio
- consumo

### 7.3 Imagen y texto

Tambien puede trabajar con:

- imagenes
- texto
- audio
- series temporales

Pero la idea de fondo sigue siendo la misma: entrada, calculo, salida y correccion.

---

## 8. Que arquitectura elegir al empezar

Reglas simples:

- si tienes numeros en una tabla: empieza con una red densa sencilla
- si tienes imagenes: piensa en CNN
- si tienes texto: piensa en Transformers
- si quieres algo basico para aprender: empieza con una red pequena y facil de visualizar

---

## 9. Que significa overfitting

Overfitting significa que la red memoriza demasiado el entrenamiento y luego falla con datos nuevos.

Es como aprenderte un examen de memoria en vez de entender el tema.

---

## 10. Regla rapida para no perderse

Si una explicacion tecnica te parece complicada, vuelve siempre a estas cinco preguntas:

1. que dato entra
2. que cuenta hace la red
3. que resultado da
4. cuanto se equivoca
5. como se corrige

Si entiendes esas cinco cosas, ya entiendes la base de una red neuronal.

---

## 11. Idea final

Una red neuronal no es magia.

Es un sistema que ajusta muchos numeros para transformar entradas en salidas cada vez mejor.

La parte tecnica puede hacerse muy compleja, pero la base siempre es la misma:

- probar
- medir error
- corregir
- repetir
