# ============================================================
# IA para Científicos Sociales - UCU
# Laboratorio 2: Exploración avanzada y comparación de modelos
#
# Autor: Danilo Freire
# Fecha: abril de 2026
#
# Este script contiene todo el código del Laboratorio 2.
# Ejecuten cada sección en orden dentro de RStudio.
# ============================================================


# --- Parte 1: Configuración y exploración --------------------

# Instalar paquetes (solo la primera vez)
# install.packages(c("tidymodels", "tidyverse", "rpart"))

# Cargar paquetes
library(tidymodels)
library(tidyverse)

# Configurar tema de ggplot
theme_set(theme_minimal(base_size = 14))

# Cargar datos
satisfaccion <- read_csv("datos/satisfaccion_democracia.csv")

# Ver estructura
glimpse(satisfaccion)


# --- Visualización base --------------------------------------

# Boxplots por satisfacción
satisfaccion |>
  select(satisfecho, confianza_gobierno, participacion_politica, edad) |>
  pivot_longer(-satisfecho, names_to = "variable", values_to = "valor") |>
  ggplot(aes(x = satisfecho, y = valor, fill = satisfecho)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~variable, scales = "free_y") +
  scale_fill_manual(values = c("#e74c3c", "#27ae60")) +
  labs(title = "Variables numéricas por nivel de satisfacción") +
  theme(legend.position = "none")


# --- Ejercicio 1: Exploración --------------------------------

# 1. Dimensiones y NAs
dim(satisfaccion)
sum(is.na(satisfaccion))

# 2. Distribución del outcome
satisfaccion |>
  count(satisfecho) |>
  mutate(prop = round(n / sum(n), 3))

# 3. Elijan una pregunta y grafíquenla.
#    Ejemplo: satisfacción por zona
satisfaccion |>
  count(zona, satisfecho) |>
  group_by(zona) |>
  mutate(prop = n / sum(n)) |>
  ggplot(aes(x = zona, y = prop, fill = satisfecho)) +
  geom_col(position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("#e74c3c", "#27ae60")) +
  labs(title = "Satisfacción por zona", y = "Proporción")

# Más ejemplos (país, género, ingreso vs. educación) en el apéndice.


# --- Parte 2: Feature engineering ----------------------------

# Convertir variables categóricas a factores
satisfaccion <- satisfaccion |>
  mutate(
    satisfecho = factor(satisfecho, levels = c("no", "si")),
    zona = factor(zona),
    genero = factor(genero),
    pais = factor(pais)
  )

# Crear nuevas variables base
satisfaccion <- satisfaccion |>
  mutate(
    grupo_edad = cut(edad,
                     breaks = c(0, 30, 50, 70, 100),
                     labels = c("joven", "adulto", "mayor", "anciano")),
    ingreso_percapita = ingreso_hogar / 3.5,
    alta_participacion = if_else(participacion_politica > 50, "alta", "baja")
  )


# --- Ejercicio 2: Crear un feature ---------------------------

# Elijan UNA variable derivada para crear.
# Ejemplo: educación alta
satisfaccion <- satisfaccion |>
  mutate(
    educacion_alta = if_else(educacion_anos > 12, "alta", "baja")
  )

# Otras opciones (pueden hacer una distinta):
# grupo_ingreso = cut(ingreso_hogar,
#                     breaks = quantile(ingreso_hogar, c(0, 1/3, 2/3, 1)),
#                     labels = c("bajo", "medio", "alto"),
#                     include.lowest = TRUE)
# noticias_alto = if_else(consumo_noticias > 5, "alto", "bajo")


# --- Parte 3: División y modelos -----------------------------

# Dividir datos
set.seed(2026)
datos_split <- initial_split(satisfaccion, prop = 0.75, strata = satisfecho)
datos_train <- training(datos_split)
datos_test <- testing(datos_split)

# Folds para validación cruzada
folds <- vfold_cv(datos_train, v = 5, strata = satisfecho)

cat("Train:", nrow(datos_train), "| Test:", nrow(datos_test))


# --- Definir modelos -----------------------------------------

# 1. Regresión logística (lineal)
modelo_logistico <- logistic_reg() |>
  set_engine("glm") |>
  set_mode("classification")

# 2. Árbol de decisión (no lineal)
modelo_arbol <- decision_tree() |>
  set_engine("rpart") |>
  set_mode("classification")


# --- Fórmula base --------------------------------------------

formula_basica <- satisfecho ~ edad + educacion_anos + ingreso_hogar +
                               confianza_gobierno + consumo_noticias +
                               participacion_politica + zona


# --- Función para evaluar ------------------------------------

# event_level = "second" porque "si" es el segundo nivel del factor
evaluar_modelo <- function(modelo, formula, folds, nombre = "modelo") {
  resultados <- fit_resamples(
    modelo,
    formula,
    resamples = folds,
    metrics = metric_set(accuracy, precision, recall, roc_auc),
    control = control_resamples(event_level = "second")
  )

  collect_metrics(resultados) |>
    mutate(modelo = nombre)
}


# --- Evaluar los modelos con la fórmula base ------------------

eval_logistico <- evaluar_modelo(modelo_logistico, formula_basica, folds, "Logístico")
eval_arbol     <- evaluar_modelo(modelo_arbol,     formula_basica, folds, "Árbol")

resultados <- bind_rows(eval_logistico, eval_arbol)

# Tabla
resultados |>
  select(modelo, .metric, mean, std_err) |>
  pivot_wider(names_from = .metric, values_from = c(mean, std_err))

# Gráfico
resultados |>
  ggplot(aes(x = modelo, y = mean, fill = modelo)) +
  geom_col(alpha = 0.8) +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err),
                width = 0.2) +
  facet_wrap(~.metric, scales = "free_y") +
  scale_fill_manual(values = c("#2d4563", "#27ae60")) +
  labs(title = "Comparación de modelos", y = "Valor", x = "") +
  theme(legend.position = "none")


# --- Ejercicio 3: ¿Ayuda el feature engineering? --------------

# Re-crear folds con la columna nueva disponible
set.seed(2026)
split_ext <- initial_split(satisfaccion, prop = 0.75, strata = satisfecho)
train_ext <- training(split_ext)
folds_ext <- vfold_cv(train_ext, v = 5, strata = satisfecho)

# Fórmula extendida con el feature que crearon
formula_ext <- satisfecho ~ edad + educacion_anos + ingreso_hogar +
                            confianza_gobierno + consumo_noticias +
                            participacion_politica + zona + educacion_alta

eval_log_ext <- evaluar_modelo(modelo_logistico, formula_ext, folds_ext, "Logístico (ext)")
eval_arb_ext <- evaluar_modelo(modelo_arbol,     formula_ext, folds_ext, "Árbol (ext)")

bind_rows(eval_logistico, eval_log_ext, eval_arbol, eval_arb_ext) |>
  filter(.metric == "accuracy") |>
  select(modelo, mean, std_err) |>
  arrange(desc(mean))


# --- Modelo final en datos de test ---------------------------

ajuste_final <- modelo_logistico |>
  fit(formula_basica, data = datos_train)

pred_test <- ajuste_final |>
  predict(datos_test, type = "prob") |>
  bind_cols(predict(ajuste_final, datos_test)) |>
  bind_cols(datos_test)

# Métricas finales
pred_test |>
  metrics(truth = satisfecho, estimate = .pred_class)

# AUC-ROC
pred_test |>
  roc_auc(truth = satisfecho, .pred_si, event_level = "second")

# Coeficientes
tidy(ajuste_final) |>
  mutate(odds_ratio = exp(estimate)) |>
  arrange(p.value)


# ============================================================
# TAREA OPCIONAL: mini-proyecto para casa
#
# Escenario: un organismo regional quiere identificar qué
# factores predicen la insatisfacción con la democracia para
# diseñar políticas públicas.
#
# Pasos sugeridos:
#   1. Elegir variables con fundamento teórico
#   2. Crear al menos una variable derivada
#   3. Entrenar y comparar dos modelos con CV
#   4. Evaluar el mejor en test e interpretar coeficientes
# ============================================================


# 1. Preparar datos con variables seleccionadas + feature nuevo
datos_proy <- satisfaccion |>
  mutate(
    confianza_baja = if_else(confianza_gobierno <= 4, "si", "no")
  ) |>
  select(satisfecho, edad, confianza_gobierno, participacion_politica,
         ingreso_hogar, zona, confianza_baja, educacion_alta)

# 2. Dividir y crear folds
set.seed(2026)
sp <- initial_split(datos_proy, prop = 0.75, strata = satisfecho)
train_p <- training(sp); test_p <- testing(sp)
folds_p <- vfold_cv(train_p, v = 5, strata = satisfecho)

# 3. Evaluar regresión logística y árbol
formula_proy <- satisfecho ~ edad + confianza_gobierno +
  participacion_politica + ingreso_hogar + zona + confianza_baja

eval_log <- evaluar_modelo(modelo_logistico, formula_proy, folds_p, "Logístico")
eval_arb <- evaluar_modelo(modelo_arbol,     formula_proy, folds_p, "Árbol")
bind_rows(eval_log, eval_arb) |>
  select(modelo, .metric, mean) |>
  pivot_wider(names_from = .metric, values_from = mean)

# 4. Ajustar el mejor en train, evaluar en test
fit_proy <- modelo_logistico |> fit(formula_proy, data = train_p)
predict(fit_proy, test_p, type = "prob") |>
  bind_cols(predict(fit_proy, test_p), test_p) |>
  metrics(truth = satisfecho, estimate = .pred_class)

# 5. Interpretar coeficientes
tidy(fit_proy) |>
  mutate(odds_ratio = exp(estimate)) |>
  arrange(p.value)
