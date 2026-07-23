# ============================================================
# IA para Científicos Sociales - UCU
# Laboratorio 4: Regresión y regularización
#
# Autor: Danilo Freire
# Fecha: mayo de 2026
#
# Este script contiene todo el código del Laboratorio 4.
# Cada sección coincide con una diapositiva (DEMO o EJERCICIO).
# La solución del Ejercicio 1 está incluida tal como aparece
# en el apéndice del .qmd.
# ============================================================


# --- Parte 1: Preparación --------------------------------------

# Si algún paquete falta: install.packages(c("tidymodels", "tidyverse", "glmnet"))
library(tidymodels)   # rsample, parsnip, recipes, workflows, tune, yardstick
library(tidyverse)    # incluye readr (read_csv), dplyr, ggplot2, etc.
library(glmnet)       # motor para LASSO, Ridge, Elastic Net

set.seed(2026)

# Cargar el dataset
datos <- read_csv("datos/latinobarometro_sim.csv", show_col_types = FALSE)
# Lean el archivo directo desde la web:
# datos <- read_csv("https://raw.githubusercontent.com/danilofreire/introduccion-ia-ucu/main/clases/dia-02/datos/latinobarometro_sim.csv", show_col_types = FALSE)

# Convertir categóricas a factor
datos <- datos |>
  mutate(
    pais         = factor(pais),
    zona         = factor(zona),
    genero       = factor(genero),
    uso_internet = factor(uso_internet, levels = c("nunca", "semanal", "diario"))
  )

glimpse(datos)


# --- Dividir los datos (75/25) ---------------------------------

# Estratificada por el outcome (con una variable numérica,
# rsample estratifica por cuartiles)
datos_split <- initial_split(datos, prop = 0.75, strata = satisfaccion_vida)
datos_train <- training(datos_split)
datos_test  <- testing(datos_split)

cat("Train:", nrow(datos_train), "obs | Test:", nrow(datos_test), "obs\n")


# --- Receta de preprocesamiento --------------------------------

# Predecir con todas las variables menos voto (outcome del Lab 3) y pais (alta cardinalidad)
receta <- recipe(satisfaccion_vida ~ ., data = datos_train) |>
  step_rm(pais, voto) |>                       # eliminar variables no predictivas o con demasiados niveles
  step_dummy(all_nominal_predictors()) |>      # categóricas → dummies
  step_normalize(all_numeric_predictors()) |>  # normalizar (importante para regularización)
  step_zv(all_predictors())                    # eliminar varianza cero

receta |> prep() |> juice() |> glimpse()


# --- Parte 2: OLS baseline -------------------------------------

# Modelo + workflow + ajuste (en una pipeline)
modelo_ols <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")

ajuste_ols <- workflow() |>
  add_recipe(receta) |>
  add_model(modelo_ols) |>
  fit(data = datos_train)

# Coeficientes ordenados por magnitud (top 5)
tidy(ajuste_ols) |>
  arrange(desc(abs(estimate))) |>
  head(5)


# --- Evaluar OLS en test ---------------------------------------

# Generar predicciones en test
pred_ols <- augment(ajuste_ols, datos_test)

# Métricas (rmse, rsq, mae)
metricas_ols <- pred_ols |>
  metrics(truth = satisfaccion_vida, estimate = .pred)

metricas_ols


# --- Parte 3: LASSO con tuning ---------------------------------

# Definir LASSO con penalty a ajustar
# tune() = "este valor se busca con CV"
# mixture = 1 → LASSO puro (L1); 0 sería Ridge
modelo_lasso <- linear_reg(penalty = tune(), mixture = 1) |>
  set_engine("glmnet") |>
  set_mode("regression")

wf_lasso <- workflow() |> add_recipe(receta) |> add_model(modelo_lasso)

# Grilla de 30 valores de λ en escala logarítmica (10^-4 a 10^0)
grilla_lambda <- grid_regular(penalty(range = c(-4, 0)), levels = 30)

head(grilla_lambda)


# --- Tuning de LASSO con CV ------------------------------------

# Crear folds (10-fold CV)
folds <- vfold_cv(datos_train, v = 10)

# Tuning: probar cada valor de la grilla con CV
resultados_lasso <- tune_grid(
  wf_lasso,
  resamples = folds,
  grid      = grilla_lambda,
  metrics   = metric_set(rmse)
)

# Top 5 mejores λ por RMSE
resultados_lasso |>
  collect_metrics() |>
  filter(.metric == "rmse") |>
  arrange(mean) |>
  head(5)


# --- Visualizar tuning de LASSO --------------------------------

autoplot(resultados_lasso) +
  scale_x_log10() +
  theme_minimal() +
  labs(title = "RMSE vs. λ (escala log)")


# --- Seleccionar λ y ajuste final ------------------------------

# 1. Mejor λ
lambda_min <- select_best(resultados_lasso, metric = "rmse")

# 2. Finalizar workflow con ese λ y ajustar a todo el train
ajuste_lasso <- wf_lasso |>
  finalize_workflow(lambda_min) |>
  fit(data = datos_train)

# Métricas en test (las reutilizamos en la tabla comparativa)
pred_lasso     <- augment(ajuste_lasso, datos_test)
metricas_lasso <- pred_lasso |>
  metrics(truth = satisfaccion_vida, estimate = .pred)

# 3. Coeficientes y conteo de variables eliminadas
coef_lasso <- tidy(ajuste_lasso) |>
  filter(term != "(Intercept)") |>
  arrange(desc(abs(estimate)))

cat("Variables eliminadas (coef = 0):",
    sum(coef_lasso$estimate == 0), "de", nrow(coef_lasso), "\n")
coef_lasso |> head(8)


# --- Comparar coeficientes OLS vs. LASSO ----------------------

coef_ols  <- tidy(ajuste_ols)   |> filter(term != "(Intercept)") |>
             select(term, OLS   = estimate)
coef_lasc <- tidy(ajuste_lasso) |> filter(term != "(Intercept)") |>
             select(term, LASSO = estimate)

left_join(coef_ols, coef_lasc, by = "term") |>
  pivot_longer(c(OLS, LASSO), names_to = "modelo", values_to = "coef") |>
  ggplot(aes(x = reorder(term, abs(coef)), y = coef, fill = modelo)) +
  geom_col(position = "dodge") + coord_flip() +
  scale_fill_manual(values = c("OLS" = "#3498DB", "LASSO" = "#E74C3C")) +
  labs(title = "Coeficientes: OLS vs. LASSO",
       x = NULL, y = "Coeficiente (normalizado)") +
  theme_minimal()


# --- Parte 4: Ridge por analogía -------------------------------

# --- Ejercicio 1: Implementar Ridge (solución) -----------------

# Cambio único respecto a LASSO: mixture = 0
modelo_ridge <- linear_reg(penalty = tune(), mixture = 0) |>
  set_engine("glmnet") |>
  set_mode("regression")

wf_ridge <- workflow() |> add_recipe(receta) |> add_model(modelo_ridge)

resultados_ridge <- tune_grid(
  wf_ridge, resamples = folds, grid = grilla_lambda,
  metrics = metric_set(rmse)
)

lambda_ridge <- select_best(resultados_ridge, metric = "rmse")
ajuste_ridge <- finalize_workflow(wf_ridge, lambda_ridge) |>
  fit(data = datos_train)

pred_ridge     <- augment(ajuste_ridge, datos_test)
metricas_ridge <- pred_ridge |>
  metrics(truth = satisfaccion_vida, estimate = .pred)

# Ridge nunca elimina variables (puede mostrar 1e-10 = prácticamente 0 pero no exacto)
cat("Variables eliminadas en Ridge:",
    sum(tidy(ajuste_ridge)$estimate[-1] == 0), "\n")
metricas_ridge


# --- Parte 5: Comparación final --------------------------------

# Tabla comparativa de los 3 modelos
tabla_final <- bind_rows(
  metricas_ols   |> mutate(modelo = "OLS"),
  metricas_lasso |> mutate(modelo = "LASSO"),
  metricas_ridge |> mutate(modelo = "Ridge")
) |>
  select(modelo, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  arrange(rmse)

tabla_final


# --- Ejercicio 2: ¿Predicción o explicación? -------------------

# Ejercicio de discusión (no requiere código nuevo). Miren la tabla
# comparativa y el gráfico de coeficientes OLS vs. LASSO y respondan:
# - ¿Qué variables redujo LASSO a cero? ¿Tiene sentido sustantivo?
# - ¿Explicar o predecir? Para explicar conviene un modelo simple
#   (OLS o LASSO); para predecir, elijan por RMSE en test.
# - ¿Por qué las diferencias de RMSE son tan pequeñas con estos datos?
#   (relaciones casi lineales y n mayor que el número de predictores)


# ============================================================
# MATERIAL OPCIONAL (apéndices del laboratorio)
# Estas secciones no se cubren en clase. Se incluyen para
# quienes quieran profundizar en casa.
# ============================================================


# --- Apéndice 3: Dos criterios para elegir λ ------------------

# Criterio 1: mínimo RMSE (el que usamos en el lab)
lambda_min <- select_best(resultados_lasso, metric = "rmse")

# Criterio 2: regla de un error estándar (modelo más parsimonioso)
lambda_1se <- select_by_one_std_err(resultados_lasso, metric = "rmse", desc(penalty))

bind_rows(
  lambda_min |> mutate(criterio = "min"),
  lambda_1se |> mutate(criterio = "1-SE")
)


# --- Apéndice 4: Capstone — outcome diferente -----------------

# Aplicar el mismo pipeline a `satisfaccion_democracia` (escala 1-5)
# en vez de `satisfaccion_vida`. ¿Cómo cambian los resultados?

# 1. Nueva receta para predecir satisfaccion_democracia
receta_dem <- recipe(satisfaccion_democracia ~ ., data = datos_train) |>
  step_rm(pais, voto, satisfaccion_vida) |>      # excluir outcomes ajenos
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_zv(all_predictors())

# 2. LASSO con la nueva receta (mismo patrón)
wf_lasso_dem <- workflow() |> add_recipe(receta_dem) |> add_model(modelo_lasso)

resultados_dem <- tune_grid(
  wf_lasso_dem, resamples = folds, grid = grilla_lambda,
  metrics = metric_set(rmse)
)

lambda_dem <- select_best(resultados_dem, metric = "rmse")
ajuste_dem <- finalize_workflow(wf_lasso_dem, lambda_dem) |>
  fit(data = datos_train)

# 3. Evaluación
pred_dem <- augment(ajuste_dem, datos_test)

pred_dem |> metrics(truth = satisfaccion_democracia, estimate = .pred)

# Preguntas para reflexionar:
# - ¿Las mismas variables son predictivas, o cambian?
# - ¿El RMSE es mayor o menor que para satisfaccion_vida?
# - Pueden repetir con percepcion_economia como tercer outcome


# --- Apéndice 5: Elastic Net (opcional) -----------------------

# Elastic Net combina LASSO y Ridge: mixture entre 0 y 1
# Ajustar AMBOS hiperparámetros simultáneamente

modelo_enet <- linear_reg(penalty = tune(), mixture = tune()) |>
  set_engine("glmnet") |>
  set_mode("regression")

wf_enet <- workflow() |> add_recipe(receta) |> add_model(modelo_enet)

# Grilla 2D: 20 valores de λ × 5 de mixture (como en la clase)
grilla_enet <- grid_regular(
  penalty(range = c(-4, 0)),
  mixture(range = c(0, 1)),
  levels = c(20, 5)
)

resultados_enet <- tune_grid(
  wf_enet, resamples = folds, grid = grilla_enet,
  metrics = metric_set(rmse)
)

mejor_enet <- select_best(resultados_enet, metric = "rmse")
mejor_enet  # vean el mixture óptimo

ajuste_enet <- finalize_workflow(wf_enet, mejor_enet) |>
  fit(data = datos_train)

augment(ajuste_enet, datos_test) |>
  metrics(truth = satisfaccion_vida, estimate = .pred)

# Si mixture óptimo está cerca de 1 → LASSO ganó
# Si está cerca de 0 → Ridge ganó
# Si intermedio → Elastic Net mejora sobre ambos
