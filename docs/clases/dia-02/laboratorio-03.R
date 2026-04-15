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
# install.packages(c("tidyverse", "tidymodels", "ranger", "vip", "pdp", "xgboost"))

# Cargar paquetes
# require(): intenta cargar el paquete y devuelve TRUE/FALSE
# Si no está instalado (FALSE), lo instala y carga
paquetes <- c("tidyverse", "tidymodels", "ranger", "vip", "pdp", "xgboost")

for (pkg in paquetes) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Cargar tidymodels (carga varios paquetes a la vez:
# rsample, parsnip, recipes, workflows, tune, yardstick, etc.)
library(tidymodels)
library(ranger)      # Motor rápido para Random Forest
library(vip)         # Variable Importance Plots
library(pdp)         # Partial Dependence Plots

# Configurar semilla para reproducibilidad
set.seed(2026)

# Cargar el dataset (generado con datos/crear-datos.R)
# show_col_types = FALSE: no mostrar los tipos detectados automáticamente
datos <- read_csv("datos/latinobarometro_sim.csv", show_col_types = FALSE)

# Convertir variables categóricas a factores
# factor(): convierte texto en variable categórica con niveles ordenados
# levels = c(...): define el orden de las categorías
datos <- datos |>
  mutate(
    pais = factor(pais),
    zona = factor(zona),
    genero = factor(genero),
    uso_internet = factor(uso_internet, levels = c("nunca", "semanal", "diario")),
    # levels: el primer nivel es la clase positiva (evento) para yardstick
    voto = factor(voto, levels = c("si", "no"))
  )

# Exploración inicial
# glimpse(): muestra las columnas, tipos y primeros valores de forma compacta
glimpse(datos)

# Distribución del voto
datos |>
  count(voto) |>
  mutate(proporcion = n / sum(n))

# Visualización
ggplot(datos, aes(x = voto, fill = voto)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +
  scale_fill_manual(values = c("no" = "#E74C3C", "si" = "#27AE60")) +
  labs(title = "Distribución de la participación electoral",
       x = "¿Votó en la última elección?", y = "Frecuencia") +
  theme_minimal() +
  theme(legend.position = "none")

# initial_split(): divide los datos aleatoriamente en train y test
# prop = 0.75: 75% para entrenamiento, 25% para test
# strata = voto: estratificar para mantener la proporción de clases
division <- initial_split(datos, prop = 0.75, strata = voto)

datos_train <- training(division)
datos_test <- testing(division)

# Verificar proporciones
cat("Proporción en train:\n")
prop.table(table(datos_train$voto))

cat("\nProporción en test:\n")
prop.table(table(datos_test$voto))


# --- Preprocesamiento con recipes ---------------------------

# recipe(): define el preprocesamiento como una "receta de cocina"
# voto ~ ...: la fórmula indica variable objetivo ~ predictores
receta <- recipe(voto ~ edad + educacion_anios + ingreso_hogar + zona +
                 genero + confianza_gobierno + confianza_justicia +
                 satisfaccion_democracia + percepcion_economia +
                 uso_internet + interes_politica,
                 data = datos_train) |>
  # step_dummy(): convierte categóricas a variables indicadoras (0/1)
  step_dummy(all_nominal_predictors()) |>
  # step_normalize(): centra (media = 0) y escala (sd = 1) las numéricas
  step_normalize(all_numeric_predictors()) |>
  # step_zv(): elimina variables con varianza cero (constantes)
  step_zv(all_predictors())

# prep(): estima los parámetros de la receta (ej: medias para normalizar)
# juice(): aplica la receta y devuelve los datos transformados
receta |> prep() |> juice() |> glimpse()


# --- Modelo baseline: Regresión logística -------------------

# logistic_reg(): modelo de regresión logística
# set_engine("glm"): usar el motor glm de R base
# set_mode("classification"): tarea de clasificación (no regresión)
modelo_logit <- logistic_reg() |>
  set_engine("glm") |>
  set_mode("classification")

# workflow(): combina preprocesamiento (receta) + modelo en un solo objeto
wf_logit <- workflow() |>
  add_recipe(receta) |>    # agregar la receta de preprocesamiento
  add_model(modelo_logit)  # agregar el modelo

# fit(): ajustar el workflow completo a los datos de entrenamiento
ajuste_logit <- fit(wf_logit, data = datos_train)

# predict(): genera predicciones de clase ("si"/"no")
# type = "prob": genera probabilidades para cada clase (.pred_no, .pred_si)
# bind_cols(): une las columnas de predicciones con los datos reales
pred_logit <- predict(ajuste_logit, datos_test) |>
  bind_cols(predict(ajuste_logit, datos_test, type = "prob")) |>
  bind_cols(datos_test |> select(voto))

# metrics(): calcula múltiples métricas de evaluación
# truth: variable real, estimate: predicción de clase, .pred_si: probabilidades
metricas_logit <- pred_logit |>
  metrics(truth = voto, estimate = .pred_class, .pred_si)

metricas_logit

# conf_mat(): construye la tabla de predicho vs. real
# autoplot(type = "heatmap"): visualización como mapa de calor
conf_mat(pred_logit, truth = voto, estimate = .pred_class) |>
  autoplot(type = "heatmap") +
  scale_fill_gradient(low = "white", high = "#2d4563") +
  labs(title = "Matriz de confusión - Regresión logística")


# --- Ejercicio 1: Threshold óptimo -------------------------

# 1. Por defecto, clasificamos como "sí" si P(sí) > 0.5
# 2. Pero este threshold puede no ser óptimo
# 3. Calcular el F1-score para distintos thresholds
# 4. ¿Cuál threshold maximiza el F1-score?

# Probar tres thresholds diferentes
for (t in c(0.3, 0.5, 0.7)) {
  pred_nuevo <- pred_logit |>
    mutate(.pred_class_nuevo = factor(
      ifelse(.pred_si > t, "si", "no"),
      levels = c("si", "no")))

  f1 <- f_meas(pred_nuevo, truth = voto,
               estimate = .pred_class_nuevo)
  cat("Threshold:", t, "- F1:", round(f1$.estimate, 3), "\n")
}

# ... probar más thresholds y visualizar


# --- Parte 2: Random Forest con tuning ---------------------

# rand_forest(): modelo de Random Forest
# tune(): marcador especial que indica "optimizar este valor automáticamente"
modelo_rf_tune <- rand_forest(
  mtry = tune(),       # Número de variables a considerar en cada split
  trees = 500,         # Número de árboles (fijo, no se optimiza)
  min_n = tune()       # Mínimo de observaciones en nodo terminal
) |>
  # importance = "impurity": calcular importancia con reducción de impureza (Gini)
  set_engine("ranger", importance = "impurity") |>
  set_mode("classification")

# workflow(): combina receta + modelo
wf_rf_tune <- workflow() |>
  add_recipe(receta) |>
  add_model(modelo_rf_tune)

# extract_parameter_set_dials(): muestra los hiperparámetros marcados con tune()
modelo_rf_tune |> extract_parameter_set_dials()

# grid_regular(): crea una grilla con valores uniformemente espaciados
# Cada parámetro tiene un rango definido; levels indica cuántos valores probar
grilla_rf <- grid_regular(
  mtry(range = c(2, 8)),    # De 2 a 8 variables por split
  min_n(range = c(5, 30)),  # De 5 a 30 obs mínimas por nodo
  levels = c(4, 4)          # 4 valores de cada uno = 4 x 4 = 16 combinaciones
)

# Ver la grilla
grilla_rf

# Alternativa: grilla aleatoria (más eficiente para muchos hiperparámetros)
# grilla_rf <- grid_random(
#   mtry(range = c(2, 8)),
#   min_n(range = c(5, 30)),
#   size = 20  # 20 combinaciones aleatorias
# )

# vfold_cv(): divide los datos en v grupos (folds) para validación cruzada
# strata = voto: estratificar para que cada fold mantenga la proporción de clases
folds <- vfold_cv(datos_train, v = 5, strata = voto)

# Ver los folds
folds

# tune_grid(): evalúa cada combinación de hiperparámetros con CV
# Para cada combinación de la grilla, entrena el modelo en cada fold
# y calcula las métricas especificadas
resultados_tune <- tune_grid(
  wf_rf_tune,               # el workflow con tune() pendientes
  resamples = folds,         # los folds de validación cruzada
  grid = grilla_rf,          # la grilla de combinaciones a probar
  # metric_set(): define qué métricas calcular en cada evaluación
  metrics = metric_set(accuracy, roc_auc, f_meas),
  control = control_grid(verbose = FALSE)  # no imprimir progreso
)

# collect_metrics(): extrae los resultados promediados de todos los folds
resultados_tune |>
  collect_metrics() |>
  filter(.metric == "roc_auc") |>
  arrange(desc(mean))

# autoplot(): método genérico que sabe graficar objetos de tidymodels
autoplot(resultados_tune) +
  theme_minimal() +
  labs(title = "Resultados del tuning de Random Forest")


# --- Seleccionar el mejor modelo ----------------------------

# select_best(): elige la combinación con el mejor valor de la métrica
mejor_rf <- select_best(resultados_tune, metric = "roc_auc")
mejor_rf

# select_by_one_std_err(): elige el modelo más simple cuyo rendimiento
# esté dentro de 1 error estándar del mejor (regla del "1-SE")
# desc(min_n): entre los candidatos, preferir más obs por nodo (más simple)
mejor_rf_1se <- select_by_one_std_err(
  resultados_tune,
  metric = "roc_auc",
  desc(min_n)
)
mejor_rf_1se

# finalize_workflow(): reemplaza los tune() del workflow por los valores
# óptimos encontrados, dejando un workflow listo para ajustar
wf_rf_final <- finalize_workflow(wf_rf_tune, mejor_rf)

# Ajustar con todos los datos de training
ajuste_rf <- fit(wf_rf_final, data = datos_train)

# Predicciones en test
pred_rf <- predict(ajuste_rf, datos_test) |>
  bind_cols(predict(ajuste_rf, datos_test, type = "prob")) |>
  bind_cols(datos_test |> select(voto))

# Comparar con baseline
metricas_rf <- pred_rf |>
  metrics(truth = voto, estimate = .pred_class, .pred_si)

# Mostrar comparación
cat("Regresión logística:\n")
print(metricas_logit)
cat("\nRandom Forest:\n")
print(metricas_rf)


# --- Curva ROC comparativa ----------------------------------

# roc_curve(): calcula sensibilidad y especificidad para cada umbral
# truth: la variable real, .pred_si: probabilidades predichas
roc_logit <- pred_logit |>
  roc_curve(truth = voto, .pred_si) |>
  mutate(modelo = "Regresión logística")

roc_rf <- pred_rf |>
  roc_curve(truth = voto, .pred_si) |>
  mutate(modelo = "Random Forest")

# Combinar y graficar
bind_rows(roc_logit, roc_rf) |>
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = modelo)) +
  geom_path(linewidth = 1.2) +
  geom_abline(linetype = "dashed", color = "gray50") +
  coord_equal() +
  labs(title = "Comparación de curvas ROC",
       x = "1 - Especificidad (Tasa de falsos positivos)",
       y = "Sensibilidad (Tasa de verdaderos positivos)",
       color = "Modelo") +
  theme_minimal()


# --- Parte 3: Interpretación --------------------------------

# extract_fit_parsnip(): extrae el modelo ajustado del workflow
# (devuelve un objeto parsnip, no el objeto nativo del motor)
modelo_extraido <- extract_fit_parsnip(ajuste_rf)

# vip(): gráfico de importancia de variables
# num_features: cuántas variables mostrar (las más importantes)
vip(modelo_extraido, num_features = 15) +
  labs(title = "Importancia de variables (Gini)",
       subtitle = "Random Forest para predicción de voto") +
  theme_minimal()

# extract_fit_engine(): extrae el objeto nativo del motor (ranger)
# (a diferencia de extract_fit_parsnip que devuelve un objeto parsnip)
modelo_ranger <- extract_fit_engine(ajuste_rf)

# bake(): aplica una receta ya estimada (prep) a datos nuevos o existentes
datos_prep <- bake(prep(receta), new_data = datos_train)

# partial(): calcula el Partial Dependence Plot
# pred.var: la variable para la que se calcula el efecto marginal
# prob = TRUE: trabajar con probabilidades (no clases)
# which.class = 1: calcular para la primera clase ("si")
pdp_edad <- partial(
  modelo_ranger,
  pred.var = "edad",
  train = datos_prep,
  prob = TRUE,
  which.class = 1  # Clase "si" (primer nivel del factor)
)

# Graficar
autoplot(pdp_edad) +
  labs(title = "Partial Dependence Plot: Edad",
       subtitle = "Efecto marginal sobre P(voto = sí)",
       x = "Edad (normalizada)", y = "Probabilidad predicha de votar") +
  theme_minimal()

# PDP para interés político
pdp_interes <- partial(
  modelo_ranger,
  pred.var = "interes_politica",
  train = datos_prep,
  prob = TRUE,
  which.class = 1  # Clase "si"
)

autoplot(pdp_interes) +
  labs(title = "Partial Dependence Plot: Interés en política",
       subtitle = "Efecto marginal sobre P(voto = sí)",
       x = "Interés en política (normalizado)",
       y = "Probabilidad predicha de votar") +
  theme_minimal()


# --- Ejercicio 2: Análisis por país -------------------------

# 1. Filtrar los datos para Uruguay solamente
# 2. Entrenar el mismo modelo de Random Forest (sin tuning, con hiperparámetros fijos)
# 3. ¿Cambia la importancia de las variables?
# 4. ¿La edad sigue siendo el predictor más importante?

datos_uruguay <- datos |>
  filter(pais == "Uruguay")

# Crear receta y workflow, ajustar el modelo...
# Comparar el VIP con el modelo general


# --- Parte 4: Comparación de modelos -----------------------

# boost_tree(): modelo de Gradient Boosting (construye árboles secuencialmente)
# trees: número total de árboles a construir uno tras otro
# tree_depth: profundidad máxima de cada árbol (a optimizar)
# learn_rate: tasa de aprendizaje, controla cuánto aporta cada árbol nuevo
# min_n: mínimo de observaciones por nodo terminal (fijo)
modelo_xgb <- boost_tree(
  trees = 500,
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = 10
) |>
  set_engine("xgboost") |>
  set_mode("classification")

# Workflow: receta + modelo
wf_xgb <- workflow() |>
  add_recipe(receta) |>
  add_model(modelo_xgb)

# Grilla de búsqueda para XGBoost
# learn_rate(range = c(-3, -1)): en escala log10, es decir 10^-3=0.001 a 10^-1=0.1
grilla_xgb <- grid_regular(
  tree_depth(range = c(3, 8)),
  learn_rate(range = c(-3, -1)),
  levels = c(3, 3)
)

# Tuning con validación cruzada
resultados_xgb <- tune_grid(
  wf_xgb,
  resamples = folds,
  grid = grilla_xgb,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = FALSE)
)

# Seleccionar, finalizar y ajustar el mejor modelo
mejor_xgb <- select_best(resultados_xgb, metric = "roc_auc")
wf_xgb_final <- finalize_workflow(wf_xgb, mejor_xgb)
ajuste_xgb <- fit(wf_xgb_final, data = datos_train)

# Predicciones de todos los modelos
pred_xgb <- predict(ajuste_xgb, datos_test) |>
  bind_cols(predict(ajuste_xgb, datos_test, type = "prob")) |>
  bind_cols(datos_test |> select(voto))

# Calcular métricas para todos
metricas_todos <- bind_rows(
  pred_logit |> metrics(truth = voto, estimate = .pred_class, .pred_si) |>
    mutate(modelo = "Regresión logística"),
  pred_rf |> metrics(truth = voto, estimate = .pred_class, .pred_si) |>
    mutate(modelo = "Random Forest"),
  pred_xgb |> metrics(truth = voto, estimate = .pred_class, .pred_si) |>
    mutate(modelo = "XGBoost")
)

# Tabla resumen
metricas_todos |>
  select(modelo, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  arrange(desc(roc_auc))


# --- Curvas ROC de todos los modelos ------------------------

roc_xgb <- pred_xgb |>
  roc_curve(truth = voto, .pred_si) |>
  mutate(modelo = "XGBoost")

bind_rows(roc_logit, roc_rf, roc_xgb) |>
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = modelo)) +
  geom_path(linewidth = 1.2) +
  geom_abline(linetype = "dashed", color = "gray50") +
  coord_equal() +
  scale_color_manual(values = c("Regresión logística" = "#3498DB",
                                "Random Forest" = "#27AE60",
                                "XGBoost" = "#9B59B6")) +
  labs(title = "Comparación de curvas ROC",
       x = "1 - Especificidad", y = "Sensibilidad") +
  theme_minimal()


# --- Ejercicio 3: Ajustar un árbol de decisión -------------

# 1. Crear un modelo decision_tree() con costo de complejidad a ajustar
# 2. Usar validación cruzada para encontrar el mejor valor
# 3. Comparar con Random Forest: ¿cuánto se pierde en rendimiento?

modelo_arbol <- decision_tree(
  cost_complexity = tune(),
  tree_depth = 10,
  min_n = 10
) |>
  set_engine("rpart") |>
  set_mode("classification")

# ... completar el workflow y tuning
