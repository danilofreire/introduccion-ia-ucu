# ============================================================
# IA para Científicos Sociales - UCU
# Laboratorio 2: Exploración avanzada y comparación de modelos
#
# Autor: Danilo Freire
# Fecha: mayo de 2026
#
# Este script contiene todo el código del Laboratorio 2.
# Ejecuten cada sección en orden dentro de RStudio.
# ============================================================


# --- Parte 1: Nuevo dataset ----------------------------------

# Cargar paquetes
library(tidymodels)
library(tidyverse)
library(rpart)  # Motor de árbol de decisión

# Tema de ggplot
theme_set(theme_minimal(base_size = 14))

# Cargar el dataset
satisfaccion <- read_csv("datos/satisfaccion_democracia.csv")
glimpse(satisfaccion)


# --- Estadísticas descriptivas -------------------------------

# Resumen rápido de las variables numéricas
satisfaccion |> select(where(is.numeric)) |> summary()

# Medias por nivel de satisfacción (predictores clave)
satisfaccion |>
  group_by(satisfecho) |>
  summarise(
    confianza_media     = mean(confianza_gobierno),
    educacion_media     = mean(educacion_anos),
    ingreso_mediano     = median(ingreso_hogar),
    participacion_media = mean(participacion_politica)
  )


# --- Ejercicio 1: Inténtenlo ustedes -------------------------
#
# (Borren este bloque cuando tengan su respuesta)
#
# 1. ¿Cuántas observaciones? ¿Hay valores faltantes?
# 2. ¿Cómo se distribuye la variable objetivo `satisfecho`?
# 3. Elijan una pregunta y respondan con código o gráfico:
#    - ¿Cómo varía la satisfacción por zona o por país?
#    - ¿Qué relación hay entre ingreso y educación?
#
# Pistas en los Apéndices 1-4 (al final del script).


# --- Parte 2: Feature engineering ----------------------------

# Convertir variables categóricas a factores
satisfaccion <- satisfaccion |>
  mutate(
    satisfecho = factor(satisfecho, levels = c("no", "si")),
    zona       = factor(zona),
    genero     = factor(genero),
    pais       = factor(pais)
  )

# Crear nuevas variables
satisfaccion <- satisfaccion |>
  mutate(
    # Grupos de edad
    grupo_edad = cut(edad,
                     breaks = c(0, 30, 50, 70, 100),
                     labels = c("joven", "adulto", "mayor", "anciano")),

    # Ingreso per cápita (hogar promedio de 3.5 personas)
    ingreso_percapita = ingreso_hogar / 3.5,

    # Indicador de alta participación
    alta_participacion = if_else(participacion_politica > 50, "alta", "baja")
  )

satisfaccion |>
  select(grupo_edad, ingreso_percapita, alta_participacion) |>
  head()


# --- Ejercicio 2: Crear un feature ---------------------------
#
# (Borren este bloque cuando tengan su respuesta)
#
# Elijan UNA variable derivada y créenla. Opciones sugeridas:
# - Grupos de ingreso por terciles (ingreso_hogar)
# - Consumidor alto de noticias: > 5 horas (consumo_noticias)
# - Baja confianza en el gobierno: <= 4 (confianza_gobierno)
# - Interacción zona × género (zona, genero)
#
# Ejemplo:
# satisfaccion <- satisfaccion |>
#   mutate(educacion_alta = if_else(educacion_anos > 12, "alta", "baja"))
#
# Más ejemplos en el Apéndice 5.


# --- Dividir los datos ---------------------------------------

set.seed(2026)
datos_split <- initial_split(satisfaccion, prop = 0.75, strata = satisfecho)
datos_train <- training(datos_split)
datos_test  <- testing(datos_split)

# Folds para validación cruzada (5 particiones estratificadas)
folds <- vfold_cv(datos_train, v = 5, strata = satisfecho)

cat("Train:", nrow(datos_train), "| Test:", nrow(datos_test))


# --- Parte 3: Comparación de modelos -------------------------

# Especificación de modelos (parsnip)
modelo_logistico <- logistic_reg() |>
  set_engine("glm") |>
  set_mode("classification")

modelo_arbol <- decision_tree() |>
  set_engine("rpart") |>
  set_mode("classification")

# Fórmula con variables originales
formula_basica <- satisfecho ~ edad + educacion_anos + ingreso_hogar +
                               confianza_gobierno + consumo_noticias +
                               participacion_politica + zona


# --- Función para evaluar un modelo --------------------------

evaluar_modelo <- function(modelo, formula, folds, nombre = "modelo") {
  # 1. fit_resamples ajusta el modelo en cada fold de la validación cruzada
  resultados <- fit_resamples(
    modelo,
    formula,
    resamples = folds,
    metrics  = metric_set(accuracy, precision, recall, roc_auc),  # 2. métricas
    control  = control_resamples(event_level = "second")          # 3. "si" es la clase positiva
  )
  # 4. collect_metrics() devuelve la media y el error estándar de cada métrica
  collect_metrics(resultados) |>
    mutate(modelo = nombre)
}


# --- Evaluar ambos modelos -----------------------------------

eval_logistico <- evaluar_modelo(modelo_logistico, formula_basica, folds, "Logístico")
eval_arbol     <- evaluar_modelo(modelo_arbol,     formula_basica, folds, "Árbol")

resultados <- bind_rows(eval_logistico, eval_arbol)

# Comparación en tabla
resultados |>
  select(modelo, .metric, mean, std_err) |>
  pivot_wider(names_from = .metric, values_from = c(mean, std_err))


# --- Ejercicio 3: ¿Ayuda el feature engineering? -------------
#
# (Borren este bloque cuando tengan su respuesta)
#
# Agreguen la variable derivada que crearon antes a la fórmula
# y re-evalúen los dos modelos. Comparen con los resultados originales.
#
# 1. ¿Mejora la regresión logística? ¿Y el árbol?
# 2. ¿Cuál de los dos modelos es más sensible al feature nuevo?
#
# Solución completa en el Apéndice 6.


# --- Modelo final en datos de test ---------------------------

# Entrenar el modelo elegido con todos los datos de train
ajuste_final <- modelo_logistico |>
  fit(formula_basica, data = datos_train)

# augment() junta predicciones (clases + probabilidades) con los datos
# originales en una sola línea. Equivale a hacer dos predict() + bind_cols().
# Añade las columnas .pred_class, .pred_no y .pred_si.
pred_test <- ajuste_final |> augment(datos_test)

# Métricas finales y AUC-ROC
pred_test |> metrics(truth = satisfecho, estimate = .pred_class)
pred_test |> roc_auc(truth = satisfecho, .pred_si, event_level = "second")


# ============================================================
# MATERIAL OPCIONAL (apéndices del laboratorio)
# Estas secciones expanden conceptos mencionados en clase.
# ============================================================


# --- Apéndice 1: Exploración inicial -------------------------

dim(satisfaccion)

satisfaccion |>
  count(satisfecho) |>
  mutate(prop = round(n / sum(n), 3))

colSums(is.na(satisfaccion))

satisfaccion |>
  select(where(is.numeric)) |>
  summary()

satisfaccion |> count(pais, sort = TRUE)
satisfaccion |> count(zona)


# --- Apéndice 2: Estadísticas por variable -------------------

# Estadísticas detalladas de una variable numérica
satisfaccion |>
  summarise(
    media   = mean(ingreso_hogar),
    mediana = median(ingreso_hogar),
    sd      = sd(ingreso_hogar),
    p25     = quantile(ingreso_hogar, 0.25),
    p75     = quantile(ingreso_hogar, 0.75),
    min     = min(ingreso_hogar),
    max     = max(ingreso_hogar)
  )

# Variables categóricas: conteos y proporciones
satisfaccion |> count(zona)   |> mutate(prop = n / sum(n))
satisfaccion |> count(genero) |> mutate(prop = n / sum(n))
satisfaccion |> count(pais, sort = TRUE)

# Histograma de una variable numérica
ggplot(satisfaccion, aes(x = ingreso_hogar)) +
  geom_histogram(bins = 30, fill = "#2d4563")

# Densidad (versión suavizada del histograma)
ggplot(satisfaccion, aes(x = confianza_gobierno)) +
  geom_density(fill = "#2d4563", alpha = 0.5)


# --- Apéndice 3: Relaciones con la satisfacción --------------

# Numéricas vs satisfacción: medias por grupo
satisfaccion |>
  group_by(satisfecho) |>
  summarise(
    edad          = mean(edad),
    educacion     = mean(educacion_anos),
    ingreso       = median(ingreso_hogar),
    confianza     = mean(confianza_gobierno),
    consumo       = mean(consumo_noticias),
    participacion = mean(participacion_politica)
  )

# Correlaciones entre todas las variables numéricas
satisfaccion |>
  select(where(is.numeric)) |>
  cor() |>
  round(2)

# Categórica vs satisfacción: tabla cruzada
satisfaccion |>
  count(zona, satisfecho) |>
  group_by(zona) |>
  mutate(prop = round(n / sum(n), 3))

# Diferencia de medias (numérica vs categórica)
t.test(confianza_gobierno ~ satisfecho, data = satisfaccion)

# Independencia (categórica vs categórica)
chisq.test(satisfaccion$zona, satisfaccion$satisfecho)


# --- Apéndice 4: Más visualizaciones -------------------------

# Boxplots: variables numéricas por nivel de satisfacción
satisfaccion |>
  select(satisfecho, confianza_gobierno, participacion_politica, edad) |>
  pivot_longer(-satisfecho, names_to = "variable", values_to = "valor") |>
  ggplot(aes(x = satisfecho, y = valor, fill = satisfecho)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~variable, scales = "free_y") +
  scale_fill_manual(values = c("#e74c3c", "#27ae60")) +
  labs(title = "Variables numéricas por nivel de satisfacción") +
  theme(legend.position = "none")

# Satisfacción por zona
satisfaccion |>
  count(zona, satisfecho) |>
  group_by(zona) |>
  mutate(prop = n / sum(n)) |>
  ggplot(aes(x = zona, y = prop, fill = satisfecho)) +
  geom_col(position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("#e74c3c", "#27ae60")) +
  labs(title = "Satisfacción por zona", y = "Proporción")

# Satisfacción por país
satisfaccion |>
  count(pais, satisfecho) |>
  group_by(pais) |>
  mutate(prop = n / sum(n)) |>
  filter(satisfecho == "si") |>
  ggplot(aes(x = reorder(pais, prop), y = prop)) +
  geom_col(fill = "#27ae60", alpha = 0.8) +
  coord_flip() +
  labs(title = "Proporción de satisfechos por país", x = "", y = "Proporción")

# Satisfacción por género
satisfaccion |>
  count(genero, satisfecho) |>
  group_by(genero) |>
  mutate(prop = n / sum(n)) |>
  ggplot(aes(x = genero, y = prop, fill = satisfecho)) +
  geom_col(position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("#e74c3c", "#27ae60")) +
  labs(title = "Satisfacción por género", y = "Proporción")

# Ingreso vs educación
ggplot(satisfaccion, aes(x = educacion_anos, y = ingreso_hogar, color = satisfecho)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE) +
  scale_color_manual(values = c("#e74c3c", "#27ae60")) +
  labs(title = "Ingreso vs. educación por satisfacción")


# --- Apéndice 5: Crear más features --------------------------

satisfaccion <- satisfaccion |>
  mutate(
    # 1. Grupos de ingreso por terciles
    grupo_ingreso = cut(ingreso_hogar,
                        breaks = quantile(ingreso_hogar, c(0, 1/3, 2/3, 1)),
                        labels = c("bajo", "medio", "alto"),
                        include.lowest = TRUE),

    # 2. Consumidor alto de noticias
    noticias_alto = if_else(consumo_noticias > 5, "alto", "bajo"),

    # 3. Baja confianza en el gobierno
    confianza_baja = if_else(confianza_gobierno <= 4, "si", "no"),

    # 4. Interacción zona + género
    zona_genero = paste(zona, genero, sep = "_")
  )

satisfaccion |> count(grupo_ingreso)
satisfaccion |> count(noticias_alto)
satisfaccion |> count(confianza_baja)
satisfaccion |> count(zona_genero)


# --- Apéndice 6: ¿Ayuda el feature engineering? --------------

# Crear feature derivado
satisfaccion <- satisfaccion |>
  mutate(educacion_alta = if_else(educacion_anos > 12, "alta", "baja"),
         educacion_alta = factor(educacion_alta))

# Re-crear folds con la columna nueva disponible
set.seed(2026)
split_ext <- initial_split(satisfaccion, prop = 0.75, strata = satisfecho)
train_ext <- training(split_ext)
folds_ext <- vfold_cv(train_ext, v = 5, strata = satisfecho)

# Fórmula extendida
formula_ext <- satisfecho ~ edad + educacion_anos + ingreso_hogar +
                            confianza_gobierno + consumo_noticias +
                            participacion_politica + zona + educacion_alta

# Evaluar los dos modelos con el feature nuevo
eval_log_ext <- evaluar_modelo(modelo_logistico, formula_ext, folds_ext, "Logístico (ext)")
eval_arb_ext <- evaluar_modelo(modelo_arbol,     formula_ext, folds_ext, "Árbol (ext)")

bind_rows(eval_logistico, eval_log_ext, eval_arbol, eval_arb_ext) |>
  filter(.metric == "accuracy") |>
  select(modelo, mean, std_err) |>
  arrange(desc(mean))

# La mejora por agregar una variable derivada suele ser modesta.
# El feature engineering rinde cuando captura relaciones no lineales o
# interacciones que el modelo no puede descubrir por sí solo.


# --- Apéndice 7: Visualizar la comparación de modelos --------

# Versión visual de la comparación de modelos:
# barras = media de cada métrica, líneas verticales = error estándar.
resultados |>
  ggplot(aes(x = modelo, y = mean, fill = modelo)) +
  geom_col(alpha = 0.8) +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err),
                width = 0.2) +
  facet_wrap(~.metric, scales = "free_y") +
  scale_fill_manual(values = c("#2d4563", "#27ae60")) +
  labs(title = "Comparación de modelos", y = "Valor", x = "") +
  theme(legend.position = "none")
