# ============================================================
# IA para Científicos Sociales - UCU
# Laboratorio 1: Primer flujo de trabajo con tidymodels
#
# Autor: Danilo Freire
# Fecha: mayo de 2026
#
# Este script contiene todo el código del Laboratorio 1.
# Ejecuten cada sección en orden dentro de RStudio.
# ============================================================


# --- Parte 1: Configuración y exploración --------------------

# Instalar paquetes (solo la primera vez)
# install.packages(c("tidymodels", "tidyverse"))

# Cargar paquetes
library(tidymodels)
library(tidyverse)

# Cargar los datos
datos <- read_csv("datos/indicadores_mundiales.csv")
# Leé el archivo directo desde la web:
# datos <- read_csv("https://raw.githubusercontent.com/danilofreire/introduccion-ia-ucu/main/clases/dia-01/datos/indicadores_mundiales.csv")

# Ver las primeras filas
glimpse(datos)

# Resumen estadístico
summary(datos)

# Distribución del outcome
datos |>
  count(crecimiento_alto) |>
  mutate(prop = n / sum(n))


# --- Ejercicio 1: Inténtenlo ustedes --------------------------
#
# (Borren este bloque cuando tengan su respuesta)
#
# 1. ¿Cuántas observaciones y variables tiene el dataset?
# 2. ¿Hay valores faltantes (NA)?
# 3. ¿Cómo se distribuye cada variable numérica?
# 4. ¿Hay correlaciones fuertes entre las variables?
#
# Pistas: dim(), glimpse(), nrow(), ncol(), sum(is.na()),
#         colSums(is.na()), summary(), cor(), select(where(is.numeric))


# --- Parte 2: Preprocesamiento y división --------------------

# Convertir el outcome a factor y descartar columnas no numéricas.
# pais tiene 179 valores únicos y rompería el modelo;
# continente lo dejamos fuera para usar solo predictores numéricos.
datos_modelo <- datos |>
  mutate(crecimiento_alto = factor(crecimiento_alto, levels = c("no", "si"))) |>
  select(-pais, -continente)

glimpse(datos_modelo)

# Ver los niveles del factor
# (el primer nivel es la clase "negativa", el segundo la "positiva")
levels(datos_modelo$crecimiento_alto)

# Dividir: 75% entrenamiento, 25% prueba
set.seed(2026)
datos_split <- initial_split(datos_modelo, prop = 0.75, strata = crecimiento_alto)

datos_train <- training(datos_split)
datos_test  <- testing(datos_split)

cat("Entrenamiento:", nrow(datos_train), "filas\n")
cat("Prueba:", nrow(datos_test), "filas\n")

# Verificar la estratificación: las proporciones deben ser similares
datos_train |>
  count(crecimiento_alto) |>
  mutate(prop = round(n / sum(n), 3))

datos_test |>
  count(crecimiento_alto) |>
  mutate(prop = round(n / sum(n), 3))


# --- Parte 3: Entrenamiento y evaluación ---------------------

# Especificar el modelo (todavía no entrena, solo declara)
modelo_log <- logistic_reg() |>
  set_engine("glm") |>
  set_mode("classification")

modelo_log

# Ajustar el modelo a los datos de entrenamiento
ajuste <- modelo_log |>
  fit(crecimiento_alto ~ ., data = datos_train)

# Ver los coeficientes en escala log-odds
# (la interpretación como odds ratios está en el Apéndice 7)
tidy(ajuste)

# Predecir clases en los datos de prueba.
# .pred_class es la columna que tidymodels crea por ti;
# el punto al principio evita chocar con columnas del usuario.
predicciones <- ajuste |>
  predict(datos_test) |>
  bind_cols(datos_test)

predicciones |>
  select(crecimiento_alto, .pred_class) |>
  head(8)

# Matriz de confusión
conf_mat(predicciones, truth = crecimiento_alto, estimate = .pred_class)

# Conjunto completo de métricas.
# event_level = "second" hace que precision y recall se calculen
# tomando "si" como clase positiva (sin esto, se usa "no").
#
# Recordatorio:
# - accuracy:        de todas las predicciones, cuántas acertó
# - precision (ppv): de las que predijo "si", cuántas eran "si"
# - recall (sens):   de los "si" reales, cuántos detectó
predicciones |>
  conf_mat(truth = crecimiento_alto, estimate = .pred_class) |>
  summary(event_level = "second")


# --- Ejercicio 2: Interpretación ------------------------------
#
# Discutan en grupos de 2-3 personas durante 3-5 minutos:
#
# 1. ¿El modelo tiene mejor precisión o mejor recall?
# 2. ¿Qué significa eso en términos prácticos?
#    - Si predecimos "crecimiento alto" cuando no lo es, ¿qué pasa?
#    - Si no detectamos un caso de crecimiento alto, ¿qué pasa?
# 3. ¿Qué métrica priorizarían ustedes y por qué?
#
# La reflexión completa está en el Apéndice 2.


# --- Parte 4: Más allá de la clasificación binaria -----------

# Probabilidades en lugar de clases.
# .pred_no y .pred_si son las probabilidades por nivel del factor.
pred_probs <- ajuste |>
  predict(datos_test, type = "prob") |>
  bind_cols(datos_test)

pred_probs |>
  select(crecimiento_alto, .pred_no, .pred_si) |>
  head(5)

# Curva ROC y AUC.
# event_level = "second" fija "si" como el evento positivo
# (es el segundo nivel del factor; sin esto el AUC sale invertido).
pred_probs |>
  roc_curve(truth = crecimiento_alto, .pred_si, event_level = "second") |>
  autoplot()

pred_probs |>
  roc_auc(truth = crecimiento_alto, .pred_si, event_level = "second")


# ============================================================
# MATERIAL OPCIONAL (apéndices del laboratorio)
# Estas secciones expanden conceptos mencionados en clase.
# La validación cruzada aparece formalmente en el Laboratorio 2.
# ============================================================

# --- Apéndice 1: Solución del Ejercicio 1 ---------------------

# 1. Dimensiones del dataset
dim(datos)
nrow(datos)
ncol(datos)

# 2. Valores faltantes
sum(is.na(datos))
colSums(is.na(datos))

# 3. Distribución de variables numéricas
datos |>
  select(where(is.numeric)) |>
  summary()

# 4. Correlaciones entre variables
datos |>
  select(where(is.numeric)) |>
  cor() |>
  round(2)


# --- Apéndice 2: Precisión y recall por separado -------------

predicciones |>
  precision(truth = crecimiento_alto,
            estimate = .pred_class,
            event_level = "second")

predicciones |>
  recall(truth = crecimiento_alto,
         estimate = .pred_class,
         event_level = "second")

# O ambas a la vez con event_level dentro de summary()
predicciones |>
  conf_mat(truth = crecimiento_alto, estimate = .pred_class) |>
  summary(event_level = "second")


# --- Apéndice 3: Observaciones inciertas ---------------------

# Observaciones con probabilidad cercana a 0.5: las más difíciles
# de clasificar. En contextos reales merecen revisión manual.
pred_probs |>
  select(crecimiento_alto, .pred_no, .pred_si) |>
  mutate(incertidumbre = abs(.pred_si - 0.5)) |>
  filter(incertidumbre < 0.1) |>
  arrange(incertidumbre)


# --- Apéndice 4: Comparar precisión y recall por umbral ------

# Para cada umbral, recalculamos la clase predicha y medimos:
# - umbral bajo  -> recall alto, precisión baja
# - umbral alto  -> precisión alta, recall bajo

# 1. Tres copias de las predicciones, cada una con su umbral
pred_umbrales <- bind_rows(
  pred_probs |> mutate(umbral = 0.3),
  pred_probs |> mutate(umbral = 0.5),
  pred_probs |> mutate(umbral = 0.7)
) |>
  # 2. Clasificar como "si" si la probabilidad supera el umbral de esa copia
  mutate(
    clase_predicha = if_else(.pred_si >= umbral, "si", "no"),
    clase_predicha = factor(clase_predicha, levels = c("no", "si"))
  )

# 3. Calcular precisión y recall dentro de cada umbral
pred_umbrales |>
  group_by(umbral) |>
  summarise(
    precision = precision_vec(crecimiento_alto, clase_predicha,
                              event_level = "second"),
    recall = recall_vec(crecimiento_alto, clase_predicha,
                        event_level = "second")
  )


# --- Apéndice 5: Validación cruzada --------------------------

# Crear 5 folds estratificados
set.seed(2026)
folds <- vfold_cv(datos_train, v = 5, strata = crecimiento_alto)
folds

# Ajustar el modelo en cada fold
cv_results <- modelo_log |>
  fit_resamples(
    crecimiento_alto ~ .,
    resamples = folds,
    metrics = metric_set(accuracy, roc_auc),
    control = control_resamples(event_level = "second")
  )

# collect_metrics() devuelve la media y el error estándar por métrica
collect_metrics(cv_results)

# Comparar con las métricas de la división única
predicciones |>
  metrics(truth = crecimiento_alto, estimate = .pred_class)


# --- Apéndice 7: Interpretar los coeficientes ----------------
# (El Apéndice 6 de las diapositivas es solo la tabla de
#  interpretación del AUC, sin código nuevo.)

# Los coeficientes de la regresión logística están en escala de
# log-odds. Para leerlos como odds ratios: exp(coeficiente).
# Un coeficiente positivo aumenta la probabilidad de "si";
# uno negativo la disminuye.
tidy(ajuste) |>
  mutate(odds_ratio = exp(estimate)) |>
  select(term, estimate, odds_ratio, p.value)
