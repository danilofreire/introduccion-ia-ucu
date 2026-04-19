# ============================================================
# IA para Científicos Sociales - UCU
# Laboratorio 4: Regresión y regularización
#
# Autor: Danilo Freire
# Fecha: abril de 2026
#
# Este script contiene todo el código del Laboratorio 4.
# Ejecuten cada sección en orden dentro de RStudio.
# ============================================================


# --- Parte 1: Preparación y baseline -------------------------

# Paquetes necesarios
paquetes <- c("tidyverse", "tidymodels", "glmnet", "ranger", "vip")

for (pkg in paquetes) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

library(tidymodels)
library(glmnet)   # Motor para LASSO, Ridge, Elastic Net

set.seed(2026)

# Cargar el dataset
datos <- read_csv("datos/latinobarometro_sim.csv", show_col_types = FALSE)
glimpse(datos)

datos <- datos |>
  mutate(
    pais = factor(pais),
    zona = factor(zona),
    genero = factor(genero),
    uso_internet = factor(uso_internet, levels = c("nunca", "semanal", "diario"))
  )

summary(datos$satisfaccion_vida)

ggplot(datos, aes(x = satisfaccion_vida)) +
  geom_histogram(binwidth = 1, fill = "#2d4563", color = "white") +
  labs(title = "Distribución de satisfacción con la vida",
       x = "Satisfacción (1-10)", y = "Frecuencia") +
  theme_minimal()

# División train/test
division <- initial_split(datos, prop = 0.75)
datos_train <- training(division)
datos_test  <- testing(division)

cat("Observaciones en train:", nrow(datos_train), "\n")
cat("Observaciones en test:", nrow(datos_test), "\n")

cat("\nMedia satisfacción (train):", mean(datos_train$satisfaccion_vida))
cat("\nMedia satisfacción (test):", mean(datos_test$satisfaccion_vida))

# Receta de preprocesamiento
receta <- recipe(satisfaccion_vida ~ edad + educacion_anios + ingreso_hogar +
                 zona + genero + confianza_gobierno + confianza_justicia +
                 satisfaccion_democracia + percepcion_economia +
                 uso_internet + interes_politica,
                 data = datos_train) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_zv(all_predictors())

# Verificar
receta |> prep() |> juice() |> glimpse()


# --- Modelo baseline: OLS ------------------------------------

modelo_ols <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")

wf_ols <- workflow() |>
  add_recipe(receta) |>
  add_model(modelo_ols)

ajuste_ols <- fit(wf_ols, data = datos_train)

tidy(ajuste_ols) |> arrange(desc(abs(estimate)))

pred_ols <- predict(ajuste_ols, datos_test) |>
  bind_cols(datos_test |> select(satisfaccion_vida))

metricas_ols <- pred_ols |>
  metrics(truth = satisfaccion_vida, estimate = .pred)

metricas_ols

# Visualizar predicciones vs. reales
ggplot(pred_ols, aes(x = satisfaccion_vida, y = .pred)) +
  geom_point(alpha = 0.5) +
  geom_abline(color = "red", linetype = "dashed") +
  labs(title = "Predicciones OLS vs. valores reales",
       x = "Satisfacción real", y = "Satisfacción predicha") +
  theme_minimal()


# --- Parte 2: LASSO con tuning -------------------------------

# LASSO: mixture = 1
modelo_lasso <- linear_reg(penalty = tune(), mixture = 1) |>
  set_engine("glmnet") |>
  set_mode("regression")

wf_lasso <- workflow() |>
  add_recipe(receta) |>
  add_model(modelo_lasso)

# penalty en escala log10: 10^-4 a 10^0
grilla_lambda <- grid_regular(
  penalty(range = c(-4, 0)),
  levels = 30
)

head(grilla_lambda)

# 10-fold CV
folds <- vfold_cv(datos_train, v = 10)

resultados_lasso <- tune_grid(
  wf_lasso, resamples = folds, grid = grilla_lambda,
  metrics = metric_set(rmse, rsq, mae)
)

resultados_lasso |>
  collect_metrics() |>
  filter(.metric == "rmse") |>
  arrange(mean) |>
  head(10)

autoplot(resultados_lasso) +
  scale_x_log10() +
  theme_minimal() +
  labs(title = "Tuning de LASSO: RMSE vs. penalty (λ)")


# --- Ejercicio 1: Efecto del lambda --------------------------

# Ajustar LASSO con varios valores fijos de lambda y contar
# cuántas variables elimina cada uno.
lambdas <- c(0.001, 0.01, 0.1, 0.5, 1)

resultados_lambda <- map_df(lambdas, function(l) {
  modelo <- linear_reg(penalty = l, mixture = 1) |>
    set_engine("glmnet") |>
    set_mode("regression")

  ajuste <- workflow() |>
    add_recipe(receta) |>
    add_model(modelo) |>
    fit(data = datos_train)

  coefs <- tidy(ajuste) |> filter(term != "(Intercept)")
  tibble(lambda = l,
         vars_eliminadas = sum(coefs$estimate == 0),
         vars_total = nrow(coefs))
})

resultados_lambda


# --- Seleccionar lambda y ajustar LASSO final ----------------

lambda_min <- select_best(resultados_lasso, metric = "rmse")
lambda_1se <- select_by_one_std_err(resultados_lasso, metric = "rmse",
                                    desc(penalty))

cat("Lambda mínimo:", lambda_min$penalty,
    " | Lambda 1SE:", lambda_1se$penalty, "\n")

# Usamos lambda mínimo
wf_lasso_final <- finalize_workflow(wf_lasso, lambda_min)
ajuste_lasso <- fit(wf_lasso_final, data = datos_train)

coef_lasso <- tidy(ajuste_lasso) |>
  filter(term != "(Intercept)") |>
  arrange(desc(abs(estimate)))

coef_lasso
cat("\nVariables eliminadas:", sum(coef_lasso$estimate == 0),
    "de", nrow(coef_lasso))

# Comparar coeficientes OLS vs. LASSO
coef_ols <- tidy(ajuste_ols) |>
  filter(term != "(Intercept)") |>
  select(term, estimate_ols = estimate)

coef_lasso_comp <- tidy(ajuste_lasso) |>
  filter(term != "(Intercept)") |>
  select(term, estimate_lasso = estimate)

comparacion <- left_join(coef_ols, coef_lasso_comp, by = "term") |>
  pivot_longer(cols = starts_with("estimate"),
               names_to = "modelo", values_to = "coef") |>
  mutate(modelo = ifelse(modelo == "estimate_ols", "OLS", "LASSO"))

ggplot(comparacion, aes(x = reorder(term, abs(coef)), y = coef, fill = modelo)) +
  geom_col(position = "dodge") +
  coord_flip() +
  scale_fill_manual(values = c("OLS" = "#3498DB", "LASSO" = "#E74C3C")) +
  labs(title = "Comparación de coeficientes: OLS vs. LASSO",
       x = "Variable", y = "Coeficiente (normalizado)") +
  theme_minimal()

# Evaluar LASSO en test
pred_lasso <- predict(ajuste_lasso, datos_test) |>
  bind_cols(datos_test |> select(satisfaccion_vida))

metricas_lasso <- pred_lasso |>
  metrics(truth = satisfaccion_vida, estimate = .pred)

cat("OLS:\n");   print(metricas_ols)
cat("\nLASSO:\n"); print(metricas_lasso)


# --- Parte 3: Ridge -----------------------------------------

# Ridge: mixture = 0
modelo_ridge <- linear_reg(penalty = tune(), mixture = 0) |>
  set_engine("glmnet") |>
  set_mode("regression")

wf_ridge <- workflow() |>
  add_recipe(receta) |>
  add_model(modelo_ridge)

resultados_ridge <- tune_grid(
  wf_ridge, resamples = folds, grid = grilla_lambda,
  metrics = metric_set(rmse)
)

lambda_ridge <- select_best(resultados_ridge, metric = "rmse")
ajuste_ridge <- finalize_workflow(wf_ridge, lambda_ridge) |>
  fit(data = datos_train)

pred_ridge <- predict(ajuste_ridge, datos_test) |>
  bind_cols(datos_test |> select(satisfaccion_vida))

metricas_ridge <- pred_ridge |>
  metrics(truth = satisfaccion_vida, estimate = .pred)

# Ridge no elimina variables
cat("Variables con coef = 0 en Ridge:",
    sum(tidy(ajuste_ridge)$estimate[-1] == 0), "\n\n")
metricas_ridge


# --- Parte 4: Comparación final ------------------------------

# Tabla comparativa (OLS, LASSO, Ridge)
tabla_comparacion <- bind_rows(
  metricas_ols   |> mutate(modelo = "OLS"),
  metricas_lasso |> mutate(modelo = "LASSO"),
  metricas_ridge |> mutate(modelo = "Ridge")
) |>
  select(modelo, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  arrange(rmse)

tabla_comparacion

# Visualización de predicciones
todas_pred <- bind_rows(
  pred_ols   |> mutate(modelo = "OLS"),
  pred_lasso |> mutate(modelo = "LASSO"),
  pred_ridge |> mutate(modelo = "Ridge")
)

ggplot(todas_pred, aes(x = satisfaccion_vida, y = .pred)) +
  geom_point(alpha = 0.4, color = "#2d4563") +
  geom_abline(color = "red", linetype = "dashed") +
  facet_wrap(~modelo, nrow = 1) +
  labs(title = "Predicciones vs. valores reales",
       x = "Satisfacción real", y = "Satisfacción predicha") +
  theme_minimal()


# ============================================================
# MATERIAL OPCIONAL
# Estas secciones no se cubren en clase. Se incluyen para
# quienes quieran profundizar en casa. Requieren instalar
# los paquetes ranger y, opcionalmente, vip.
# ============================================================


# --- Apéndice 2: Interacciones -------------------------------

receta_interact <- recipe(satisfaccion_vida ~ edad + educacion_anios + ingreso_hogar +
                          zona + genero + confianza_gobierno + confianza_justicia +
                          satisfaccion_democracia + percepcion_economia +
                          uso_internet + interes_politica,
                          data = datos_train) |>
  step_dummy(all_nominal_predictors()) |>
  step_interact(terms = ~ edad:educacion_anios) |>
  step_normalize(all_numeric_predictors()) |>
  step_zv(all_predictors())

wf_lasso_interact <- workflow() |>
  add_recipe(receta_interact) |>
  add_model(modelo_lasso)

resultados_interact <- tune_grid(
  wf_lasso_interact, resamples = folds,
  grid = grilla_lambda, metrics = metric_set(rmse)
)

mejor_interact <- select_best(resultados_interact, metric = "rmse")
ajuste_interact <- finalize_workflow(wf_lasso_interact, mejor_interact) |>
  fit(data = datos_train)

tidy(ajuste_interact) |>
  filter(term != "(Intercept)") |>
  arrange(desc(abs(estimate)))


# --- Apéndice 3: Países como predictores ---------------------

receta_pais <- recipe(satisfaccion_vida ~ ., data = datos_train) |>
  step_rm(voto) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_zv(all_predictors())

wf_lasso_pais <- workflow() |>
  add_recipe(receta_pais) |>
  add_model(modelo_lasso)

resultados_pais <- tune_grid(
  wf_lasso_pais, resamples = folds,
  grid = grilla_lambda, metrics = metric_set(rmse)
)

mejor_pais <- select_best(resultados_pais, metric = "rmse")
ajuste_pais <- finalize_workflow(wf_lasso_pais, mejor_pais) |>
  fit(data = datos_train)

coef_pais <- tidy(ajuste_pais) |>
  filter(str_detect(term, "pais"), estimate != 0) |>
  arrange(desc(abs(estimate)))

cat("Países con coeficiente distinto de cero:", nrow(coef_pais), "\n")
coef_pais


# --- Apéndice 4: Elastic Net ---------------------------------

modelo_enet <- linear_reg(penalty = tune(), mixture = tune()) |>
  set_engine("glmnet") |>
  set_mode("regression")

wf_enet <- workflow() |>
  add_recipe(receta) |>
  add_model(modelo_enet)

# 15 lambdas x 5 valores de mixture
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
mejor_enet

ajuste_enet <- finalize_workflow(wf_enet, mejor_enet) |>
  fit(data = datos_train)

pred_enet <- predict(ajuste_enet, datos_test) |>
  bind_cols(datos_test |> select(satisfaccion_vida))

pred_enet |> metrics(truth = satisfaccion_vida, estimate = .pred)


# --- Apéndice 5: Random Forest para regresión ----------------

modelo_rf <- rand_forest(
  trees = 500,
  mtry = tune(),
  min_n = tune()
) |>
  set_engine("ranger") |>
  set_mode("regression")

wf_rf <- workflow() |>
  add_recipe(receta) |>
  add_model(modelo_rf)

grilla_rf <- grid_regular(
  mtry(range = c(2, 8)),
  min_n(range = c(5, 20)),
  levels = c(4, 4)
)

resultados_rf <- tune_grid(
  wf_rf, resamples = folds, grid = grilla_rf,
  metrics = metric_set(rmse)
)

mejor_rf <- select_best(resultados_rf, metric = "rmse")
ajuste_rf <- finalize_workflow(wf_rf, mejor_rf) |>
  fit(data = datos_train)

pred_rf <- predict(ajuste_rf, datos_test) |>
  bind_cols(datos_test |> select(satisfaccion_vida))

pred_rf |> metrics(truth = satisfaccion_vida, estimate = .pred)
