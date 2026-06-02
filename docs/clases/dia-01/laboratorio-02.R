# ============================================================
# IA para Científicos Sociales - UCU
# Laboratorio 2: Validación cruzada y comparación de modelos
# (dataset: indicadores_mundiales.csv, mismo que el lab 1)
#
# Autor: Danilo Freire
# Fecha: mayo de 2026
#
# Este script contiene todo el código del Laboratorio 2.
# Ejecuten cada sección en orden dentro de RStudio.
# ============================================================


# --- Parte 1: Carga y exploración ----------------------------

# Cargar paquetes
library(tidymodels)
library(tidyverse)
library(rpart)  # Motor de árbol de decisión

# Tema de ggplot
theme_set(theme_minimal(base_size = 14))

# Cargar el dataset (mismo del lab 1)
datos <- read_csv("datos/indicadores_mundiales.csv")
glimpse(datos)


# --- Estadísticas descriptivas -------------------------------

# Resumen rápido de las variables numéricas
datos |> select(where(is.numeric)) |> summary()

# Medias por nivel de crecimiento (3 predictores ilustrativos; los otros
# 5 los exploran en el Ejercicio 1 y en los Apéndices 2-4)
datos |>
  group_by(crecimiento_alto) |>
  summarise(
    educacion_media     = mean(gasto_educacion),
    internet_medio      = mean(acceso_internet),
    urbanizacion_media  = mean(urbanizacion)
  )


# --- Ejercicio 1: Inténtenlo ustedes -------------------------
#
# (Borren este bloque cuando tengan su respuesta)
#
# 1. ¿Cuántas observaciones? ¿Hay valores faltantes?
# 2. ¿Cómo se distribuye la variable objetivo `crecimiento_alto`?
# 3. Elijan UNA pregunta y respondan con una comparación numérica
#    (el gráfico es opcional):
#    - Elijan una de gasto_salud, desempleo, inversion_extranjera,
#      indice_gobierno_digital y comparen su media (o mediana) entre
#      países con crecimiento_alto = "si" y "no". ¿La diferencia es grande?
#    - ¿Están altamente correlacionados acceso_internet y urbanizacion?
#      Si lo están, podríamos no necesitar a los dos en el modelo.
#
# Ejemplos:
# datos |>
#   group_by(crecimiento_alto) |>
#   summarise(media_gasto_salud = mean(gasto_salud))
#
# cor(datos$acceso_internet, datos$urbanizacion)
#
# Pistas en los Apéndices 1-4 (al final del script).


# --- Parte 2: Feature engineering ----------------------------

# Preparar los datos
# Creamos un frame nuevo (datos_modelo) listo para modelar.
# `datos` queda intacto con `pais` y `continente` para los apéndices.
datos_modelo <- datos |>
  mutate(crecimiento_alto = factor(crecimiento_alto, levels = c("no", "si"))) |>
  select(-pais, -continente)

glimpse(datos_modelo)

# Mantenemos el orden c("no", "si") para que "si" sea la clase positiva.
# Más adelante usaremos event_level = "second".


# --- Crear nuevas variables ----------------------------------

datos_modelo <- datos_modelo |>
  mutate(
    # Grupos de urbanización (rural, mixto, urbano)
    grupo_urbanizacion = cut(urbanizacion,
                             breaks = c(0, 50, 75, 100),
                             labels = c("rural", "mixto", "urbano"),
                             include.lowest = TRUE),

    # Intensidad fiscal social: PIB que va a salud + educación
    gasto_social = gasto_educacion + gasto_salud,

    # Indicador binario de alto desempleo (lo usaremos como ejemplo)
    alto_desempleo = factor(if_else(desempleo > 7, "alto", "bajo"))
  )

datos_modelo |>
  select(grupo_urbanizacion, gasto_social, alto_desempleo) |>
  head()


# --- Ejercicio 2: Crear un feature ---------------------------
#
# (Borren este bloque cuando tengan su respuesta)
#
# Elijan UNA variable derivada y créenla. Opciones sugeridas:
# - Gasto en educación alto: por encima de la mediana (gasto_educacion)
# - Acceso a internet por terciles (acceso_internet)
# - Inflación alta: > 5% (inflacion)
# - Apertura financiera: IED > mediana (inversion_extranjera)
#
# Ejemplo (internet por terciles):
# datos_modelo <- datos_modelo |>
#   mutate(internet_grupos = cut(acceso_internet,
#                                breaks = quantile(acceso_internet, c(0, 1/3, 2/3, 1)),
#                                labels = c("bajo", "medio", "alto"),
#                                include.lowest = TRUE))
#
# Más ejemplos en el Apéndice 5.


# --- Dividir los datos ---------------------------------------

set.seed(2026)
datos_split <- initial_split(datos_modelo, prop = 0.75, strata = crecimiento_alto)
datos_train <- training(datos_split)
datos_test  <- testing(datos_split)

# Folds para validación cruzada (5 particiones estratificadas)
folds <- vfold_cv(datos_train, v = 5, strata = crecimiento_alto)

cat("Train:", nrow(datos_train), "| Test:", nrow(datos_test))

# Con 179 países y 5 folds, cada partición tiene ~27 países en validación.
# Es suficiente para entrenar, pero las métricas tendrán más ruido fold-a-fold
# que en un dataset grande. La variación entre folds es informativa, no un
# defecto del modelo.


# --- Parte 3: Comparación de modelos -------------------------

# Especificación de modelos (parsnip)
modelo_logistico <- logistic_reg() |>
  set_engine("glm") |>
  set_mode("classification")

modelo_arbol <- decision_tree() |>
  set_engine("rpart") |>
  set_mode("classification")

# Fórmula con las 8 variables numéricas originales
formula_basica <- crecimiento_alto ~ gasto_educacion + acceso_internet +
                                     urbanizacion + gasto_salud + inflacion +
                                     desempleo + inversion_extranjera +
                                     indice_gobierno_digital


# --- Evaluar el modelo logístico -----------------------------

# Validación cruzada en 5 folds. Usamos precision, recall y roc_auc en
# lugar de accuracy: con clases desbalanceadas, accuracy puede engañar.
eval_logistico <- fit_resamples(
  modelo_logistico, formula_basica,
  resamples = folds,
  metrics = metric_set(precision, recall, roc_auc),
  control = control_resamples(event_level = "second")   # "si" es la clase positiva
) |>
  collect_metrics() |>
  mutate(modelo = "Logístico")

eval_logistico


# --- Evaluar el árbol y comparar -----------------------------

# El mismo bloque que antes, cambiando modelo_logistico → modelo_arbol
eval_arbol <- fit_resamples(
  modelo_arbol, formula_basica,
  resamples = folds,
  metrics = metric_set(precision, recall, roc_auc),
  control = control_resamples(event_level = "second")
) |>
  collect_metrics() |>
  mutate(modelo = "Árbol")

# Combinar y mostrar la comparación
resultados <- bind_rows(eval_logistico, eval_arbol)

resultados |>
  select(modelo, .metric, mean, std_err) |>
  pivot_wider(names_from = .metric, values_from = c(mean, std_err))


# --- ¿Qué pasa si extendemos la fórmula? ---------------------

# Ejemplo trabajado. Reemplazamos `desempleo` (continuo) por
# `alto_desempleo` (binario) y vemos qué cambia. La idea: si el
# predictor original ya captura la información, un recorte categórico
# difícilmente ayudará.
formula_ext <- crecimiento_alto ~ gasto_educacion + acceso_internet +
                                  urbanizacion + gasto_salud + inflacion +
                                  alto_desempleo + inversion_extranjera +
                                  indice_gobierno_digital

eval_log_ext <- fit_resamples(
  modelo_logistico, formula_ext, resamples = folds,
  metrics = metric_set(precision, recall, roc_auc),
  control = control_resamples(event_level = "second")
) |>
  collect_metrics() |> mutate(modelo = "Logístico (ext)")

# Comparar el logístico básico vs. el extendido
bind_rows(eval_logistico, eval_log_ext) |>
  select(modelo, .metric, mean, std_err) |>
  pivot_wider(names_from = .metric, values_from = c(mean, std_err))

# Lección: no todo feature engineering mejora el modelo. Cuando el
# predictor original ya capta la información, agregarle (o reemplazarlo
# por) un recorte categórico es redundante. Hay que medirlo, no asumirlo.


# --- Ejercicio 3: ¿Ayuda el feature engineering? -------------
#
# (Borren este bloque cuando tengan su respuesta)
#
# Agreguen LA variable derivada que crearon en el Ejercicio 2 a la
# fórmula básica y re-evalúen los dos modelos.
#
# 1. ¿Mejora la regresión logística? ¿Y el árbol?
# 2. ¿Cuál de los dos modelos es más sensible al feature nuevo?
#
# Ejemplo con internet_grupos:
# formula_estudiante <- crecimiento_alto ~ gasto_educacion + acceso_internet +
#                                          urbanizacion + gasto_salud + inflacion +
#                                          desempleo + inversion_extranjera +
#                                          indice_gobierno_digital + internet_grupos
#
# Solución trabajada en el Apéndice 6.


# --- Modelo final en datos de test ---------------------------

# Entrenamos el modelo elegido con todos los datos de train. En el lab
# usamos formula_ext para ilustrar augment(), aunque las métricas son
# casi idénticas a formula_basica.
ajuste_final <- modelo_logistico |>
  fit(formula_ext, data = datos_train)

# augment() junta predicciones (clases + probabilidades) con los datos
# originales en una sola línea. Equivale a hacer dos predict() + bind_cols().
# Añade las columnas .pred_class, .pred_no y .pred_si.
pred_test <- ajuste_final |> augment(datos_test)

# Métricas finales (las mismas que usamos en validación cruzada)
pred_test |> precision(truth = crecimiento_alto, estimate = .pred_class, event_level = "second")
pred_test |> recall(truth = crecimiento_alto, estimate = .pred_class, event_level = "second")
pred_test |> roc_auc(truth = crecimiento_alto, .pred_si, event_level = "second")


# ============================================================
# MATERIAL OPCIONAL (apéndices del laboratorio)
# Estas secciones expanden conceptos mencionados en clase.
# ============================================================


# --- Apéndice 1: Exploración inicial -------------------------

dim(datos)

datos |>
  count(crecimiento_alto) |>
  mutate(prop = round(n / sum(n), 3))

colSums(is.na(datos))

datos |>
  select(where(is.numeric)) |>
  summary()


# --- Apéndice 2: Una variable a fondo ------------------------

# Tomemos `inflacion`. En el cuerpo usamos su mediana en las estadísticas
# descriptivas porque tiene cola larga: la media (≈ 12.7) es mayor que
# la mediana (≈ 11.7), con países de hasta ~37% que estiran el promedio.

datos |>
  summarise(
    media   = mean(inflacion),
    mediana = median(inflacion),
    sd      = sd(inflacion),
    p25     = quantile(inflacion, 0.25),
    p75     = quantile(inflacion, 0.75),
    min     = min(inflacion),
    max     = max(inflacion)
  )

# Histograma (muestra la cola larga)
ggplot(datos, aes(x = inflacion)) +
  geom_histogram(bins = 30, fill = "#2d4563") +
  labs(x = "Inflación anual (%)", y = "Países")

# Densidad por grupo (¿separa crecimiento alto de bajo?)
ggplot(datos, aes(x = inflacion, fill = crecimiento_alto)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("#e74c3c", "#27ae60")) +
  labs(x = "Inflación anual (%)")

# Cambien el nombre de la variable (gasto_salud, desempleo,
# indice_gobierno_digital, etc.) para repetir el análisis con otra.
# La mayoría de los predictores son casi simétricos; inflacion es la excepción.


# --- Apéndice 3: Predictores vs. objetivo --------------------

# Medias por grupo de las variables no mostradas en el cuerpo
datos |>
  group_by(crecimiento_alto) |>
  summarise(
    salud         = mean(gasto_salud),
    inflacion     = mean(inflacion),
    desempleo     = mean(desempleo),
    ied           = mean(inversion_extranjera),
    gob_digital   = mean(indice_gobierno_digital)
  )

# Correlación entre todos los predictores numéricos
# (¿hay pares redundantes?)
datos |>
  select(where(is.numeric)) |>
  cor() |>
  round(2)

# Test formal para una variable concreta
t.test(indice_gobierno_digital ~ crecimiento_alto, data = datos)


# --- Apéndice 4: Boxplots y dispersión -----------------------

# Boxplots de las 4 variables no mostradas en el cuerpo, por nivel
# de crecimiento.
datos |>
  select(crecimiento_alto, gasto_salud, desempleo, inversion_extranjera, indice_gobierno_digital) |>
  pivot_longer(-crecimiento_alto, names_to = "variable", values_to = "valor") |>
  ggplot(aes(x = crecimiento_alto, y = valor, fill = crecimiento_alto)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~variable, scales = "free_y") +
  scale_fill_manual(values = c("#e74c3c", "#27ae60")) +
  labs(title = "Las 4 variables no mostradas, por nivel de crecimiento",
       x = NULL, y = NULL) +
  theme(legend.position = "none")

# Acceso a internet vs. urbanización (¿colinealidad?)
ggplot(datos, aes(x = urbanizacion, y = acceso_internet, color = crecimiento_alto)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE) +
  scale_color_manual(values = c("#e74c3c", "#27ae60")) +
  labs(x = "Urbanización (%)", y = "Acceso a internet (%)")

# La correlación numérica complementa el gráfico
cor(datos$urbanizacion, datos$acceso_internet)


# --- Apéndice 5: Crear más features --------------------------

# Una posible solución para cada una de las cuatro opciones del Ej. 2
datos_modelo <- datos_modelo |>
  mutate(
    # 1. Gasto en educación alto (gasto_educacion)
    educacion_alta = if_else(gasto_educacion > median(gasto_educacion), "alta", "baja"),

    # 2. Acceso a internet por terciles (acceso_internet)
    internet_grupos = cut(acceso_internet,
                          breaks = quantile(acceso_internet, c(0, 1/3, 2/3, 1)),
                          labels = c("bajo", "medio", "alto"),
                          include.lowest = TRUE),

    # 3. Inflación alta (inflacion)
    inflacion_alta = if_else(inflacion > 5, "si", "no"),

    # 4. Apertura financiera: IED por encima de la mediana (inversion_extranjera)
    apertura_financiera = if_else(inversion_extranjera > median(inversion_extranjera), "alta", "baja")
  )

datos_modelo |> count(educacion_alta)
datos_modelo |> count(internet_grupos)
datos_modelo |> count(inflacion_alta)
datos_modelo |> count(apertura_financiera)


# --- Apéndice 6: ¿Ayuda el feature engineering? --------------

# Crear el feature derivado (si no lo hicieron antes)
datos_modelo <- datos_modelo |>
  mutate(
    internet_grupos = cut(acceso_internet,
                          breaks = quantile(acceso_internet, c(0, 1/3, 2/3, 1)),
                          labels = c("bajo", "medio", "alto"),
                          include.lowest = TRUE),
    internet_grupos = factor(internet_grupos)
  )

# Re-crear folds con la columna nueva disponible
set.seed(2026)
split_int <- initial_split(datos_modelo, prop = 0.75, strata = crecimiento_alto)
train_int <- training(split_int)
folds_int <- vfold_cv(train_int, v = 5, strata = crecimiento_alto)

# Fórmula con el feature nuevo
formula_int <- crecimiento_alto ~ gasto_educacion + acceso_internet +
                                  urbanizacion + gasto_salud + inflacion +
                                  desempleo + inversion_extranjera +
                                  indice_gobierno_digital + internet_grupos

# Evaluar el logístico con folds_int (mismo patrón que el cuerpo del lab)
eval_log_int <- fit_resamples(
  modelo_logistico, formula_int, resamples = folds_int,
  metrics = metric_set(precision, recall, roc_auc),
  control = control_resamples(event_level = "second")
) |>
  collect_metrics() |> mutate(modelo = "Logístico (internet)")

# Evaluar el árbol con folds_int
eval_arb_int <- fit_resamples(
  modelo_arbol, formula_int, resamples = folds_int,
  metrics = metric_set(precision, recall, roc_auc),
  control = control_resamples(event_level = "second")
) |>
  collect_metrics() |> mutate(modelo = "Árbol (internet)")

bind_rows(eval_logistico, eval_log_int, eval_arbol, eval_arb_int) |>
  filter(.metric == "roc_auc") |>
  select(modelo, mean, std_err) |>
  arrange(desc(mean))

# La mejora por agregar una sola variable derivada suele ser modesta.
# El feature engineering rinde cuando las variables nuevas capturan
# relaciones no lineales o interacciones que el modelo no puede
# descubrir por sí solo (en un árbol esto pesa menos, porque ya captura
# no linealidades automáticamente).


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
