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
# install.packages(c("tidymodels", "tidyverse", "kknn", "rpart"))

# Cargar paquetes
library(tidymodels)
library(tidyverse)

# Configurar tema de ggplot
theme_set(theme_minimal(base_size = 14))

# Cargar datos
satisfaccion <- read_csv("datos/satisfaccion_democracia.csv")

# Ver estructura
glimpse(satisfaccion)


# --- Ejercicio 1: Exploración inicial -----------------------

# 1. Dimensiones
dim(satisfaccion)

# 2. Distribución del outcome
satisfaccion |>
  count(satisfecho) |>
  mutate(prop = round(n / sum(n), 3))

# 3. Valores faltantes
colSums(is.na(satisfaccion))

# 4. Resumen de numéricas
satisfaccion |>
  select(where(is.numeric)) |>
  summary()

# 5. Distribución por país y zona
satisfaccion |> count(pais, sort = TRUE)
satisfaccion |> count(zona)


# --- Visualizar distribuciones -------------------------------

# Histogramas de variables numéricas
satisfaccion |>
  select(where(is.numeric)) |>
  pivot_longer(everything(), names_to = "variable", values_to = "valor") |>
  ggplot(aes(x = valor)) +
  geom_histogram(bins = 30, fill = "#2d4563", alpha = 0.7) +
  facet_wrap(~variable, scales = "free") +
  labs(title = "Distribución de variables numéricas")

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


# --- Ejercicio 2: Más visualizaciones -----------------------

# 1. Satisfacción por zona
satisfaccion |>
  count(zona, satisfecho) |>
  group_by(zona) |>
  mutate(prop = n / sum(n)) |>
  ggplot(aes(x = zona, y = prop, fill = satisfecho)) +
  geom_col(position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("#e74c3c", "#27ae60")) +
  labs(title = "Satisfacción por zona", y = "Proporción")

# 2. Satisfacción por país
satisfaccion |>
  count(pais, satisfecho) |>
  group_by(pais) |>
  mutate(prop = n / sum(n)) |>
  filter(satisfecho == "si") |>
  ggplot(aes(x = reorder(pais, prop), y = prop)) +
  geom_col(fill = "#27ae60", alpha = 0.8) +
  coord_flip() +
  labs(title = "Proporción de satisfechos por país", x = "", y = "Proporción")

# 3. Satisfacción por género
satisfaccion |>
  count(genero, satisfecho) |>
  group_by(genero) |>
  mutate(prop = n / sum(n)) |>
  ggplot(aes(x = genero, y = prop, fill = satisfecho)) +
  geom_col(position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("#e74c3c", "#27ae60")) +
  labs(title = "Satisfacción por género", y = "Proporción")

# 4. Ingreso vs. educación
ggplot(satisfaccion, aes(x = educacion_anos, y = ingreso_hogar, color = satisfecho)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE) +
  scale_color_manual(values = c("#e74c3c", "#27ae60")) +
  labs(title = "Ingreso vs. educación por satisfacción")


# --- Parte 2: Feature engineering ----------------------------

# Convertir variables categóricas a factores
satisfaccion <- satisfaccion |>
  mutate(
    satisfecho = factor(satisfecho, levels = c("no", "si")),
    zona = factor(zona),
    genero = factor(genero),
    pais = factor(pais)
  )

# Crear nuevas variables
satisfaccion <- satisfaccion |>
  mutate(
    # Grupos de edad
    grupo_edad = cut(edad,
                     breaks = c(0, 30, 50, 70, 100),
                     labels = c("joven", "adulto", "mayor", "anciano")),

    # Ingreso per cápita (asumiendo hogar de 3.5 personas)
    ingreso_percapita = ingreso_hogar / 3.5,

    # Indicador de alta participación
    alta_participacion = if_else(participacion_politica > 50, "alta", "baja"),

    # Interacción: confianza × participación
    confianza_x_participacion = confianza_gobierno * participacion_politica / 100
  )


# --- Ejercicio 3: Crear más features ------------------------

satisfaccion <- satisfaccion |>
  mutate(
    # Educación alta: más de 12 años
    educacion_alta = if_else(educacion_anos > 12, "alta", "baja"),

    # Combinación zona + género
    zona_genero = paste(zona, genero, sep = "_"),

    # Grupos de ingreso (terciles)
    grupo_ingreso = cut(ingreso_hogar,
                        breaks = quantile(ingreso_hogar, c(0, 1/3, 2/3, 1)),
                        labels = c("bajo", "medio", "alto"),
                        include.lowest = TRUE),

    # Consumidor alto de noticias
    noticias_alto = if_else(consumo_noticias > 5, "alto", "bajo")
  )


# --- Parte 3: División y modelos ----------------------------

# Dividir datos
set.seed(2026)
datos_split <- initial_split(satisfaccion, prop = 0.75, strata = satisfecho)
datos_train <- training(datos_split)
datos_test <- testing(datos_split)

# Folds para validación cruzada
folds <- vfold_cv(datos_train, v = 5, strata = satisfecho)

cat("Train:", nrow(datos_train), "| Test:", nrow(datos_test))


# --- Definir modelos ----------------------------------------

# 1. Regresión logística
modelo_logistico <- logistic_reg() |>
  set_engine("glm") |>
  set_mode("classification")

# 2. Árbol de decisión
modelo_arbol <- decision_tree() |>
  set_engine("rpart") |>
  set_mode("classification")

# 3. K-Nearest Neighbors
modelo_knn <- nearest_neighbor(neighbors = 5) |>
  set_engine("kknn") |>
  set_mode("classification")


# --- Fórmulas -----------------------------------------------

# Fórmula básica
formula_basica <- satisfecho ~ edad + educacion_anos + ingreso_hogar +
                               confianza_gobierno + consumo_noticias +
                               participacion_politica + zona

# Fórmula extendida (con feature nuevo)
formula_extendida <- satisfecho ~ edad + educacion_anos + ingreso_hogar +
                                  confianza_gobierno + consumo_noticias +
                                  participacion_politica + zona +
                                  confianza_x_participacion


# --- Función para evaluar -----------------------------------

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


# --- Evaluar todos los modelos -------------------------------

eval_logistico <- evaluar_modelo(modelo_logistico, formula_basica, folds, "Logístico")
eval_arbol <- evaluar_modelo(modelo_arbol, formula_basica, folds, "Árbol")
eval_knn <- evaluar_modelo(modelo_knn, formula_basica, folds, "KNN")

# Combinar resultados
resultados <- bind_rows(eval_logistico, eval_arbol, eval_knn)

# Tabla de comparación
resultados |>
  select(modelo, .metric, mean, std_err) |>
  pivot_wider(names_from = .metric, values_from = c(mean, std_err))

# Gráfico de comparación
resultados |>
  ggplot(aes(x = modelo, y = mean, fill = modelo)) +
  geom_col(alpha = 0.8) +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err),
                width = 0.2) +
  facet_wrap(~.metric, scales = "free_y") +
  scale_fill_manual(values = c("#2d4563", "#e74c3c", "#27ae60")) +
  labs(title = "Comparación de modelos", y = "Valor", x = "") +
  theme(legend.position = "none")


# --- Ejercicio 4: Comparar con features nuevos ---------------

eval_log_ext <- evaluar_modelo(
  modelo_logistico, formula_extendida, folds, "Logístico (ext)")
eval_arbol_ext <- evaluar_modelo(
  modelo_arbol, formula_extendida, folds, "Árbol (ext)")
eval_knn_ext <- evaluar_modelo(
  modelo_knn, formula_extendida, folds, "KNN (ext)")

# Comparar accuracy: básica vs. extendida
todos <- bind_rows(
  eval_logistico, eval_arbol, eval_knn,
  eval_log_ext, eval_arbol_ext, eval_knn_ext
)

todos |>
  filter(.metric == "accuracy") |>
  select(modelo, mean, std_err) |>
  arrange(desc(mean))


# --- Modelo final en datos de test ---------------------------

# Elegir el mejor modelo (ej: regresión logística)
ajuste_final <- modelo_logistico |>
  fit(formula_basica, data = datos_train)

# Predicciones en test
pred_test <- ajuste_final |>
  predict(datos_test, type = "prob") |>
  bind_cols(predict(ajuste_final, datos_test)) |>
  bind_cols(datos_test)

# Métricas finales
pred_test |>
  metrics(truth = satisfecho, estimate = .pred_class)

# AUC-ROC (con event_level = "second")
pred_test |>
  roc_auc(truth = satisfecho, .pred_si, event_level = "second")

# Coeficientes del modelo final
tidy(ajuste_final) |>
  mutate(odds_ratio = exp(estimate)) |>
  arrange(p.value)
