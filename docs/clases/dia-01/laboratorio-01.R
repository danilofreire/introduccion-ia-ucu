# ============================================================
# IA para Científicos Sociales - UCU
# Laboratorio 1: Primer flujo de trabajo con tidymodels
#
# Autor: Danilo Freire
# Fecha: abril de 2026
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

# Ver las primeras filas
glimpse(datos)

# Resumen estadístico
summary(datos)

# Distribución del outcome
datos |>
  count(crecimiento_alto) |>
  mutate(prop = n / sum(n))


# --- Ejercicio 1: Exploración --------------------------------

# 1. Dimensiones del dataset
dim(datos)

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


# --- Parte 2: Preprocesamiento y división --------------------

# Convertir el outcome a factor
datos <- datos |>
  mutate(crecimiento_alto = factor(crecimiento_alto, levels = c("no", "si")))

# Seleccionar variables para el modelo (solo numéricas)
datos_modelo <- datos |>
  select(crecimiento_alto, gasto_educacion, acceso_internet,
         urbanizacion, gasto_salud, inflacion, desempleo,
         inversion_extranjera, indice_gobierno_digital)

glimpse(datos_modelo)

# Ver los niveles del factor
levels(datos_modelo$crecimiento_alto)

# Dividir: 75% entrenamiento, 25% prueba
set.seed(2026)
datos_split <- initial_split(datos_modelo, prop = 0.75, strata = crecimiento_alto)

datos_train <- training(datos_split)
datos_test  <- testing(datos_split)

cat("Entrenamiento:", nrow(datos_train), "filas\n")
cat("Prueba:", nrow(datos_test), "filas\n")

# Verificar estratificación
datos_train |>
  count(crecimiento_alto) |>
  mutate(prop = round(n / sum(n), 3), conjunto = "train")

datos_test |>
  count(crecimiento_alto) |>
  mutate(prop = round(n / sum(n), 3), conjunto = "test")


# --- Ejercicio 2: División de datos ---------------------------

# Pregunta 1: prop = 0.50
set.seed(2026)
split_50 <- initial_split(datos_modelo, prop = 0.50, strata = crecimiento_alto)
cat("Train:", nrow(training(split_50)), "/ Test:", nrow(testing(split_50)))

# Pregunta 2: sin estratificación
set.seed(2026)
split_sin <- initial_split(datos_modelo, prop = 0.75)
training(split_sin) |> count(crecimiento_alto) |> mutate(prop = round(n / sum(n), 3))
testing(split_sin)  |> count(crecimiento_alto) |> mutate(prop = round(n / sum(n), 3))

# Pregunta 3: diferente semilla
set.seed(999)
split_999 <- initial_split(datos_modelo, prop = 0.75, strata = crecimiento_alto)
training(split_999) |> count(crecimiento_alto) |> mutate(prop = round(n / sum(n), 3))


# --- Parte 3: Entrenamiento y evaluación ----------------------

# Especificar el modelo
modelo_log <- logistic_reg() |>
  set_engine("glm") |>
  set_mode("classification")

modelo_log

# Ajustar el modelo
ajuste <- modelo_log |>
  fit(crecimiento_alto ~ ., data = datos_train)

# Ver los coeficientes
tidy(ajuste)

# Odds ratios
tidy(ajuste) |>
  mutate(odds_ratio = exp(estimate)) |>
  select(term, estimate, odds_ratio, p.value)

# Predecir clases en datos de prueba
predicciones <- ajuste |>
  predict(datos_test) |>
  bind_cols(datos_test)

predicciones |>
  select(crecimiento_alto, .pred_class) |>
  head(10)

# Matriz de confusión
conf_mat(predicciones, truth = crecimiento_alto, estimate = .pred_class)

# Conjunto completo de métricas
predicciones |>
  conf_mat(truth = crecimiento_alto, estimate = .pred_class) |>
  summary()


# --- Ejercicio 3: Interpretación ------------------------------

# Precisión y recall
predicciones |>
  precision(truth = crecimiento_alto,
            estimate = .pred_class,
            event_level = "second")

predicciones |>
  recall(truth = crecimiento_alto,
         estimate = .pred_class,
         event_level = "second")


# --- Ejercicio 4: Probabilidades de predicción ----------------

# Obtener probabilidades
pred_probs <- ajuste |>
  predict(datos_test, type = "prob") |>
  bind_cols(datos_test)

pred_probs |>
  select(crecimiento_alto, .pred_no, .pred_si) |>
  head(10)

# Observaciones con probabilidad cercana a 0.5 (inciertas)
pred_probs |>
  select(crecimiento_alto, .pred_no, .pred_si) |>
  mutate(incertidumbre = abs(.pred_si - 0.5)) |>
  filter(incertidumbre < 0.1) |>
  arrange(incertidumbre)


# --- Ejercicio 5: Cambiar el umbral ---------------------------

# Umbral personalizado
umbral <- 0.3

pred_umbral <- pred_probs |>
  mutate(.pred_class_nuevo = factor(
    if_else(.pred_si >= umbral, "si", "no"),
    levels = c("no", "si")
  ))

pred_umbral |>
  precision(truth = crecimiento_alto, estimate = .pred_class_nuevo,
            event_level = "second")

pred_umbral |>
  recall(truth = crecimiento_alto, estimate = .pred_class_nuevo,
         event_level = "second")

# Comparar tres umbrales
purrr::map_df(c(0.3, 0.5, 0.7), function(u) {
  pred_probs |>
    mutate(.pred_u = factor(
      dplyr::if_else(.pred_si >= u, "si", "no"),
      levels = c("no", "si")
    )) |>
    summarise(
      umbral    = u,
      precision = precision_vec(crecimiento_alto, .pred_u, event_level = "second"),
      recall    = recall_vec(crecimiento_alto, .pred_u, event_level = "second")
    )
})


# --- Ejercicio 6: Validación cruzada -------------------------

set.seed(2026)
folds <- vfold_cv(datos_train, v = 5, strata = crecimiento_alto)
folds

# Ajustar modelo en cada fold
cv_results <- fit_resamples(
  modelo_log,
  crecimiento_alto ~ .,
  resamples = folds,
  metrics = metric_set(accuracy, precision, recall),
  control = control_resamples(event_level = "second")
)

# Resultados promedio
collect_metrics(cv_results)

# Comparar con la división única
predicciones |>
  metrics(truth = crecimiento_alto, estimate = .pred_class)


# --- Ejercicio 7: Curva ROC ----------------------------------

# Calcular curva ROC
# event_level = "second" indica que "si" es el evento positivo
# (el segundo nivel del factor). Sin esto, el AUC sale invertido.
roc_data <- pred_probs |>
  roc_curve(truth = crecimiento_alto, .pred_si, event_level = "second")

autoplot(roc_data)

# Calcular AUC
pred_probs |>
  roc_auc(truth = crecimiento_alto, .pred_si, event_level = "second")
