# ============================================================
# IA para Científicos Sociales - UCU
# Laboratorio 3: Clasificación avanzada
#
# Autor: Danilo Freire
# Fecha: abril de 2026
#
# Este script contiene todo el código del Laboratorio 3.
# Ejecuten cada sección en orden dentro de RStudio.
# ============================================================


# --- Parte 1: Preparación y baseline -----------------------

# Instalar paquetes (solo la primera vez)
# install.packages(c("tidyverse", "tidymodels", "ranger", "vip", "pdp", "patchwork"))

paquetes <- c("tidyverse", "tidymodels", "ranger", "vip", "pdp", "patchwork")

for (pkg in paquetes) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

library(tidymodels)
library(ranger)      # Motor rápido para Random Forest
library(vip)         # Variable Importance Plots
library(pdp)         # Partial Dependence Plots
library(patchwork)   # Combinar varios gráficos

set.seed(2026)

# Cargar los datos
datos <- read_csv("datos/latinobarometro_sim.csv", show_col_types = FALSE)

# Convertir categóricas a factores. El primer nivel de voto es
# la clase positiva para yardstick.
datos <- datos |>
  mutate(
    pais = factor(pais),
    zona = factor(zona),
    genero = factor(genero),
    uso_internet = factor(uso_internet, levels = c("nunca", "semanal", "diario")),
    voto = factor(voto, levels = c("si", "no"))
  )

# Estructura y distribución del voto
glimpse(datos)
datos |> count(voto) |> mutate(proporcion = n / sum(n))

# División train/test estratificada
division <- initial_split(datos, prop = 0.75, strata = voto)
datos_train <- training(division)
datos_test  <- testing(division)


# --- Preprocesamiento con recipes ---------------------------

receta <- recipe(voto ~ edad + educacion_anios + ingreso_hogar + zona +
                 genero + confianza_gobierno + confianza_justicia +
                 satisfaccion_democracia + percepcion_economia +
                 uso_internet + interes_politica,
                 data = datos_train) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_zv(all_predictors())


# --- Modelo baseline: Regresión logística -------------------

modelo_logit <- logistic_reg() |>
  set_engine("glm") |>
  set_mode("classification")

wf_logit <- workflow() |>
  add_recipe(receta) |>
  add_model(modelo_logit)

ajuste_logit <- fit(wf_logit, data = datos_train)

pred_logit <- predict(ajuste_logit, datos_test) |>
  bind_cols(predict(ajuste_logit, datos_test, type = "prob")) |>
  bind_cols(datos_test |> select(voto))

metricas_logit <- pred_logit |>
  metrics(truth = voto, estimate = .pred_class, .pred_si)

metricas_logit

# Matriz de confusión
conf_mat(pred_logit, truth = voto, estimate = .pred_class) |>
  autoplot(type = "heatmap") +
  scale_fill_gradient(low = "white", high = "#2d4563") +
  labs(title = "Matriz de confusión - Regresión logística")


# --- Ejercicio 1: Threshold óptimo --------------------------

# Probar varios thresholds y medir F1-score
for (t in c(0.3, 0.5, 0.7)) {
  pred_nuevo <- pred_logit |>
    mutate(.pred_class_nuevo = factor(
      ifelse(.pred_si > t, "si", "no"),
      levels = c("si", "no")))

  f1 <- f_meas(pred_nuevo, truth = voto,
               estimate = .pred_class_nuevo)
  cat("Threshold:", t, "- F1:", round(f1$.estimate, 3), "\n")
}


# --- Parte 2: Random Forest con tuning ---------------------

# mtry y min_n a optimizar; trees fijo en 500
modelo_rf_tune <- rand_forest(
  mtry = tune(),
  trees = 500,
  min_n = tune()
) |>
  set_engine("ranger", importance = "impurity") |>
  set_mode("classification")

wf_rf_tune <- workflow() |>
  add_recipe(receta) |>
  add_model(modelo_rf_tune)

# 4 x 4 = 16 combinaciones
grilla_rf <- grid_regular(
  mtry(range = c(2, 8)),
  min_n(range = c(5, 30)),
  levels = c(4, 4)
)

# 5-fold CV estratificada
folds <- vfold_cv(datos_train, v = 5, strata = voto)

resultados_tune <- tune_grid(
  wf_rf_tune,
  resamples = folds,
  grid = grilla_rf,
  metrics = metric_set(accuracy, roc_auc, f_meas),
  control = control_grid(verbose = FALSE)
)

resultados_tune |>
  collect_metrics() |>
  filter(.metric == "roc_auc") |>
  arrange(desc(mean))

autoplot(resultados_tune) +
  theme_minimal() +
  labs(title = "Resultados del tuning de Random Forest")


# --- Seleccionar y ajustar el modelo final ------------------

mejor_rf <- select_best(resultados_tune, metric = "roc_auc")
wf_rf_final <- finalize_workflow(wf_rf_tune, mejor_rf)
ajuste_rf <- fit(wf_rf_final, data = datos_train)

pred_rf <- predict(ajuste_rf, datos_test) |>
  bind_cols(predict(ajuste_rf, datos_test, type = "prob")) |>
  bind_cols(datos_test |> select(voto))

metricas_rf <- pred_rf |>
  metrics(truth = voto, estimate = .pred_class, .pred_si)

cat("Regresión logística:\n"); print(metricas_logit)
cat("\nRandom Forest:\n");     print(metricas_rf)


# --- Curva ROC comparativa ----------------------------------

roc_logit <- pred_logit |>
  roc_curve(truth = voto, .pred_si) |>
  mutate(modelo = "Regresión logística")

roc_rf <- pred_rf |>
  roc_curve(truth = voto, .pred_si) |>
  mutate(modelo = "Random Forest")

bind_rows(roc_logit, roc_rf) |>
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = modelo)) +
  geom_path(linewidth = 1.2) +
  geom_abline(linetype = "dashed", color = "gray50") +
  coord_equal() +
  labs(title = "Comparación de curvas ROC",
       x = "1 - Especificidad", y = "Sensibilidad") +
  theme_minimal()


# --- Parte 3: Interpretación --------------------------------

modelo_extraido <- extract_fit_parsnip(ajuste_rf)

vip(modelo_extraido, num_features = 15) +
  labs(title = "Importancia de variables (Gini)",
       subtitle = "Random Forest para predicción de voto") +
  theme_minimal()

# PDPs: edad e interés político (gráfico lado a lado)
modelo_ranger <- extract_fit_engine(ajuste_rf)
datos_prep <- bake(prep(receta), new_data = datos_train)

pdp_edad <- partial(modelo_ranger, pred.var = "edad",
                    train = datos_prep, prob = TRUE, which.class = 1)

pdp_interes <- partial(modelo_ranger, pred.var = "interes_politica",
                       train = datos_prep, prob = TRUE, which.class = 1)

p1 <- autoplot(pdp_edad) +
  labs(title = "PDP: Edad", x = "Edad (normalizada)",
       y = "P(voto = sí)") + theme_minimal()

p2 <- autoplot(pdp_interes) +
  labs(title = "PDP: Interés en política", x = "Interés (normalizado)",
       y = "P(voto = sí)") + theme_minimal()

p1 + p2


# --- Ejercicio 2: Análisis por país -------------------------

# Filtrar datos de Uruguay
datos_uruguay <- datos |> filter(pais == "Uruguay")
cat("Observaciones en Uruguay:", nrow(datos_uruguay), "\n")

# Con pocos datos, usamos hiperparámetros fijos en lugar de tuning
receta_uy <- recipe(voto ~ edad + educacion_anios + ingreso_hogar + zona +
                    genero + confianza_gobierno + confianza_justicia +
                    satisfaccion_democracia + percepcion_economia +
                    uso_internet + interes_politica,
                    data = datos_uruguay) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_zv(all_predictors())

modelo_rf_uy <- rand_forest(trees = 500, mtry = 4, min_n = 3) |>
  set_engine("ranger", importance = "impurity") |>
  set_mode("classification")

ajuste_rf_uy <- workflow() |>
  add_recipe(receta_uy) |>
  add_model(modelo_rf_uy) |>
  fit(data = datos_uruguay)

vip(extract_fit_parsnip(ajuste_rf_uy), num_features = 10) +
  labs(title = "Importancia de variables - solo Uruguay",
       subtitle = paste0("n = ", nrow(datos_uruguay), " observaciones")) +
  theme_minimal()


# ============================================================
# MATERIAL OPCIONAL
# Estas secciones no se cubren en clase. Se incluyen para
# quienes quieran profundizar en casa. Requieren instalar
# el paquete xgboost.
# ============================================================


# --- Opcional: Árbol de decisión ----------------------------

modelo_arbol <- decision_tree(
  cost_complexity = tune(),
  tree_depth = 10,
  min_n = 10
) |>
  set_engine("rpart") |>
  set_mode("classification")

wf_arbol <- workflow() |>
  add_recipe(receta) |>
  add_model(modelo_arbol)

# cost_complexity en escala log10: 10^-4 a 10^-1
grilla_arbol <- grid_regular(
  cost_complexity(range = c(-4, -1)),
  levels = 10
)

resultados_arbol <- tune_grid(
  wf_arbol, resamples = folds, grid = grilla_arbol,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = FALSE)
)

mejor_arbol <- select_best(resultados_arbol, metric = "roc_auc")
ajuste_arbol <- finalize_workflow(wf_arbol, mejor_arbol) |>
  fit(data = datos_train)

pred_arbol <- predict(ajuste_arbol, datos_test) |>
  bind_cols(predict(ajuste_arbol, datos_test, type = "prob")) |>
  bind_cols(datos_test |> select(voto))

cat("Árbol de decisión:\n")
print(pred_arbol |> metrics(truth = voto, estimate = .pred_class, .pred_si))


# --- Opcional: XGBoost --------------------------------------

# install.packages("xgboost")
# library(xgboost)

modelo_xgb <- boost_tree(
  trees = 500,
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = 10
) |>
  set_engine("xgboost") |>
  set_mode("classification")

wf_xgb <- workflow() |>
  add_recipe(receta) |>
  add_model(modelo_xgb)

# learn_rate en escala log10: 10^-3 a 10^-1
grilla_xgb <- grid_regular(
  tree_depth(range = c(3, 8)),
  learn_rate(range = c(-3, -1)),
  levels = c(3, 3)
)

resultados_xgb <- tune_grid(
  wf_xgb, resamples = folds, grid = grilla_xgb,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = FALSE)
)

mejor_xgb <- select_best(resultados_xgb, metric = "roc_auc")
ajuste_xgb <- finalize_workflow(wf_xgb, mejor_xgb) |>
  fit(data = datos_train)

pred_xgb <- predict(ajuste_xgb, datos_test) |>
  bind_cols(predict(ajuste_xgb, datos_test, type = "prob")) |>
  bind_cols(datos_test |> select(voto))

# Tabla comparativa con los tres modelos
bind_rows(
  pred_logit |> metrics(truth = voto, estimate = .pred_class, .pred_si) |>
    mutate(modelo = "Regresión logística"),
  pred_rf |> metrics(truth = voto, estimate = .pred_class, .pred_si) |>
    mutate(modelo = "Random Forest"),
  pred_xgb |> metrics(truth = voto, estimate = .pred_class, .pred_si) |>
    mutate(modelo = "XGBoost")
) |>
  select(modelo, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  arrange(desc(roc_auc))
