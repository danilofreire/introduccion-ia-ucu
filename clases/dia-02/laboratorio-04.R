# ============================================================
# IA para Científicos Sociales - UCU
# Laboratorio 4: Regresión y regularización
#
# Autor: Danilo Freire
# Fecha: mayo de 2026
#
# Este script contiene todo el código del Laboratorio 4.
# Cada sección coincide con una diapositiva (DEMO o EJERCICIO).
# Las soluciones de los ejercicios están incluidas tal como
# aparecen en los apéndices del .qmd.
# ============================================================


# --- Parte 1: Preparación --------------------------------------

# Si algún paquete falta: install.packages(c("glmnet", "ranger", "vip"))
library(tidymodels)   # rsample, parsnip, recipes, workflows, tune, yardstick
library(tidyverse)    # incluye readr (read_csv), dplyr, ggplot2, etc.
library(glmnet)       # motor para LASSO, Ridge, Elastic Net

set.seed(2026)

# Cargar el dataset
datos <- read_csv("datos/latinobarometro_sim.csv", show_col_types = FALSE)

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

division    <- initial_split(datos, prop = 0.75)
datos_train <- training(division)
datos_test  <- testing(division)

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

# Modelo + workflow + ajuste
modelo_ols <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")

wf_ols     <- workflow() |> add_recipe(receta) |> add_model(modelo_ols)
ajuste_ols <- fit(wf_ols, data = datos_train)

# Coeficientes ordenados por magnitud (top 5)
tidy(ajuste_ols) |>
  arrange(desc(abs(estimate))) |>
  head(5)


# --- Ejercicio 1: Evaluar OLS (solución) -----------------------

# Generar predicciones en test
pred_ols <- predict(ajuste_ols, datos_test) |>
  bind_cols(datos_test |> select(satisfaccion_vida))

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


# --- Ejercicio 2: Correr CV para LASSO (solución) --------------

# Crear folds (10-fold CV)
folds <- vfold_cv(datos_train, v = 10)

# Tuning: probar cada valor de la grilla con CV
resultados_lasso <- tune_grid(
  wf_lasso,
  resamples = folds,
  grid      = grilla_lambda,
  metrics   = metric_set(rmse, rsq)
)

# Top 10 mejores λ por RMSE
resultados_lasso |>
  collect_metrics() |>
  filter(.metric == "rmse") |>
  arrange(mean) |>
  head(10)


# --- Visualizar tuning de LASSO --------------------------------

autoplot(resultados_lasso) +
  scale_x_log10() +
  theme_minimal() +
  labs(title = "RMSE vs. λ (escala log)")


# --- Ejercicio 3: Seleccionar λ y ajustar final (solución) -----

# 1. Mejor λ
lambda_min <- select_best(resultados_lasso, metric = "rmse")
lambda_min

# 2. Finalizar workflow con ese λ y ajustar a todo el train
wf_lasso_final <- finalize_workflow(wf_lasso, lambda_min)
ajuste_lasso   <- fit(wf_lasso_final, data = datos_train)

# 3. Coeficientes y conteo de variables eliminadas
coef_lasso <- tidy(ajuste_lasso) |>
  filter(term != "(Intercept)") |>
  arrange(desc(abs(estimate)))

cat("Variables eliminadas (coef = 0):",
    sum(coef_lasso$estimate == 0), "de", nrow(coef_lasso), "\n")
coef_lasso |> head(8)

# Predicciones y métricas (para usar en Ejercicio 5)
pred_lasso     <- predict(ajuste_lasso, datos_test) |>
  bind_cols(datos_test |> select(satisfaccion_vida))
metricas_lasso <- pred_lasso |>
  metrics(truth = satisfaccion_vida, estimate = .pred)


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

# --- Ejercicio 4: Implementar Ridge (solución) -----------------

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

pred_ridge     <- predict(ajuste_ridge, datos_test) |>
  bind_cols(datos_test |> select(satisfaccion_vida))
metricas_ridge <- pred_ridge |>
  metrics(truth = satisfaccion_vida, estimate = .pred)

# Ridge nunca elimina variables (puede mostrar 1e-10 = prácticamente 0 pero no exacto)
cat("Variables eliminadas en Ridge:",
    sum(tidy(ajuste_ridge)$estimate[-1] == 0), "\n")
metricas_ridge


# --- Parte 5: Comparación final --------------------------------

# --- Ejercicio 5: Tabla comparativa (solución) -----------------

tabla_final <- bind_rows(
  metricas_ols   |> mutate(modelo = "OLS"),
  metricas_lasso |> mutate(modelo = "LASSO"),
  metricas_ridge |> mutate(modelo = "Ridge")
) |>
  select(modelo, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  arrange(rmse)

tabla_final


# ============================================================
# MATERIAL OPCIONAL (apéndices del laboratorio)
# Estas secciones no se cubren en clase. Se incluyen para
# quienes quieran profundizar en casa.
# ============================================================


# --- Apéndice 6: Capstone — outcome diferente -----------------

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
  metrics = metric_set(rmse, rsq)
)

lambda_dem <- select_best(resultados_dem, metric = "rmse")
ajuste_dem <- finalize_workflow(wf_lasso_dem, lambda_dem) |>
  fit(data = datos_train)

# 3. Evaluación
pred_dem <- predict(ajuste_dem, datos_test) |>
  bind_cols(datos_test |> select(satisfaccion_democracia))

pred_dem |> metrics(truth = satisfaccion_democracia, estimate = .pred)

# Preguntas para reflexionar:
# - ¿Las mismas variables son predictivas, o cambian?
# - ¿El RMSE es mayor o menor que para satisfaccion_vida?
# - Pueden repetir con percepcion_economia como tercer outcome


# --- Apéndice 7: Elastic Net (opcional) -----------------------

# Elastic Net combina LASSO y Ridge: mixture entre 0 y 1
# Ajustar AMBOS hiperparámetros simultáneamente

modelo_enet <- linear_reg(penalty = tune(), mixture = tune()) |>
  set_engine("glmnet") |>
  set_mode("regression")

wf_enet <- workflow() |> add_recipe(receta) |> add_model(modelo_enet)

# Grilla 2D: 15 valores de λ × 5 de mixture
grilla_enet <- grid_regular(
  penalty(range = c(-4, 0)),
  mixture(range = c(0, 1)),
  levels = c(15, 5)
)

resultados_enet <- tune_grid(
  wf_enet, resamples = folds, grid = grilla_enet,
  metrics = metric_set(rmse)
)

mejor_enet <- select_best(resultados_enet, metric = "rmse")
mejor_enet  # vean el mixture óptimo

ajuste_enet <- finalize_workflow(wf_enet, mejor_enet) |>
  fit(data = datos_train)

predict(ajuste_enet, datos_test) |>
  bind_cols(datos_test |> select(satisfaccion_vida)) |>
  metrics(truth = satisfaccion_vida, estimate = .pred)

# Si mixture óptimo está cerca de 1 → LASSO ganó
# Si está cerca de 0 → Ridge ganó
# Si intermedio → Elastic Net mejora sobre ambos
