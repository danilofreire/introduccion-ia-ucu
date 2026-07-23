# ============================================================
# IA para Científicos Sociales - UCU
# Laboratorio 9: Modelos locales y auditoría de equidad
#
# Autor: Danilo Freire
# Fecha: abril de 2026
#
# Este script contiene el código del Laboratorio 9, extraído
# de las diapositivas (19-laboratorio-09.qmd). Las soluciones
# de los ejercicios están incluidas. El bloque comentado del
# apéndice (comparación con la nube) se corre a mano si hace
# falta.
#
# Antes de empezar, instalar Ollama (https://ollama.com/) y
# descargar el modelo del laboratorio en la terminal:
#   ollama pull granite4.1:3b
# Ollama tiene que estar corriendo mientras se ejecuta este script.
#
# Requiere el dataset datos/entrevistas_confianza.csv
# ============================================================

# --- setup -----------------------------------------------------

options(htmltools.dir.version = FALSE)
library(knitr)
opts_chunk$set(
  prompt = T,
  fig.align = "center",
  dpi = 300,
  cache = T,
  message = FALSE,
  warning = FALSE,
  engine.opts = list(bash = "-l")
)

knit_hooks$set(
  prompt = function(before, options, envir) {
    options(
      prompt = if (options$engine %in% c("sh", "bash", "zsh")) "$ " else "R> ",
      continue = if (options$engine %in% c("sh", "bash", "zsh")) "$ " else "+ "
    )
  }
)

options(repos = c(CRAN = "https://cran.rstudio.com/"))
options(cli.progress_show_after = Inf)

if (!require("fontawesome", character.only = TRUE)) {
  install.packages("fontawesome", dependencies = TRUE)
  library(fontawesome, character.only = TRUE)
}

# --- conectar-desde-r ------------------------------------------

library(tidyverse)
library(ellmer)
library(quallmer)

chat <- chat_ollama(
  model = "granite4.1:3b",
  system_prompt = "Sos un asistente de investigación social. Respondé en español, en una sola oración."
)

chat$chat("¿Qué es una encuesta de victimización?")

# --- el-caso-entrevistas-que-no-pueden-salir-de-l --------------

entrevistas <- read_csv(
  "datos/entrevistas_confianza.csv",
  show_col_types = FALSE
)
# Leé el archivo directo desde la web:
# entrevistas <- read_csv("https://raw.githubusercontent.com/danilofreire/introduccion-ia-ucu/main/clases/dia-05/datos/entrevistas_confianza.csv", show_col_types = FALSE)

entrevistas |> count(motivo_humano)

# --- paso-1-definir-el-codebook --------------------------------

codebook_motivos <- qlm_codebook(
  name = "motivos_no_denuncia",
  instructions = "Leé el fragmento de entrevista. La persona fue víctima de un delito
  y no hizo la denuncia. Clasificá el MOTIVO PRINCIPAL en una de estas categorías:
  - desconfianza: no cree que la policía o la justicia investiguen o resuelvan nada
  - miedo: teme represalias del agresor o de su entorno
  - costo: el trámite implica demasiado tiempo, viajes o burocracia
  - poca_gravedad: considera que el hecho no fue suficientemente grave
  - resolucion_propia: resolvió el problema por su cuenta o por otra vía
  Elegí UNA sola categoría, la que mejor capture el motivo central del relato.",
  schema = type_object(
    motivo = type_enum(
      c("desconfianza", "miedo", "costo", "poca_gravedad", "resolucion_propia"),
      "El motivo principal por el que no denunció"
    ),
    justificacion = type_string("Una oración que justifique la categoría elegida")
  ),
  role = "Sos un investigador experto en criminología y encuestas de victimización.",
  levels = list(motivo = "nominal", justificacion = "nominal")
)

# --- paso-2-codificar-con-el-modelo-local ----------------------

codificado <- qlm_code(
  entrevistas$texto,
  codebook_motivos,
  model = "ollama/granite4.1:3b",
  max_active = 1,
  params = params(temperature = 0),
  name = "granite_local"
)

# --- inspeccionar-los-resultados -------------------------------

resultados <- as_tibble(codificado) |>
  mutate(texto = str_trunc(entrevistas$texto, 60), gold = entrevistas$motivo_humano) |>
  select(texto, motivo, gold)

resultados

# --- paso-3-validar-contra-el-gold-standard --------------------

gold <- qlm_humancoded(
  tibble(
    .id = seq_len(nrow(entrevistas)),
    motivo = entrevistas$motivo_humano
  ),
  name = "humano",
  codebook = codebook_motivos
)

validacion <- qlm_validate(
  codificado,
  gold = gold,
  by = "motivo",
  level = "nominal"
)

as_tibble(validacion) |> select(measure, value)

# --- donde-se-equivoca -----------------------------------------

resultados |> filter(motivo != gold)

# --- preparacion-del-entorno -----------------------------------

# Instalar paquetes si no están disponibles
if (!require("fairmodels")) install.packages("fairmodels")
if (!require("DALEX")) install.packages("DALEX")

# Cargar paquetes
library(tidymodels)
library(fairmodels)
library(DALEX)

# --- el-dataset-german-credit ----------------------------------

# Cargar datos incluidos en fairmodels
data("german", package = "fairmodels")

# Explorar estructura
glimpse(german)

# --- explorar-el-atributo-protegido ----------------------------

# Distribución de Risk por Sex
german |>
  count(Sex, Risk) |>
  group_by(Sex) |>
  mutate(prop = n / sum(n)) |>
  ggplot(aes(x = Sex, y = prop, fill = Risk)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = c("good" = "#2d4563", "bad" = "#e63946")) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Distribución de riesgo crediticio por sexo",
       y = "Proporción", x = "Sexo", fill = "Riesgo") +
  theme_minimal(base_size = 14)

# --- preparar-los-datos ----------------------------------------

# Dividir en train/test (estratificado por el resultado)
set.seed(123)
split <- initial_split(german, prop = 0.7, strata = Risk)
train_data <- training(split)
test_data <- testing(split)

cat("Entrenamiento:", nrow(train_data), "filas\n")
cat("Test:", nrow(test_data), "filas\n")

# --- entrenar-una-regresion-logistica --------------------------

# Resultado binario y datos sin Sex (el atributo protegido)
datos_modelo <- train_data |>
  mutate(buen_pagador = ifelse(Risk == "good", 1, 0)) |>
  select(-Risk, -Sex)

# Regresión logística: el clasificador clásico del día 2
modelo <- glm(buen_pagador ~ ., data = datos_modelo, family = binomial)

# Probabilidad de "good" en test y decisión a 0.5
pred_probs <- predict(modelo, test_data, type = "response")
pred_class <- ifelse(pred_probs > 0.5, "good", "bad")

# Accuracy global
accuracy <- mean(pred_class == test_data$Risk)
cat("Accuracy global:", round(accuracy, 3), "\n")

# --- crear-el-explicador-dalex ---------------------------------

# Crear explicador DALEX (para un glm no hace falta función a medida)
explainer <- DALEX::explain(
  model = modelo,
  data = test_data |> select(-Risk, -Sex),
  y = as.numeric(test_data$Risk == "good"),
  label = "Regresión logística",
  verbose = FALSE
)

# --- la-auditoria-fairness-check -------------------------------

# Crear objeto fairness con Sex como atributo protegido
fobject <- fairness_check(
  explainer,
  protected = test_data$Sex,
  privileged = "male",  # grupo de referencia
  cutoff = 0.5,
  verbose = FALSE
)

# Ver resumen
print(fobject)

# --- visualizar-metricas-de-equidad ----------------------------

plot(fobject)

# --- interpretar-las-metricas ----------------------------------

# Cada ratio: métrica del grupo no privilegiado ÷ la del privilegiado
fobject$fairness_check_data |>
  select(metric, score) |>
  mutate(score = round(score, 3))

# --- mitigacion-umbrales-diferenciados -------------------------

# Un umbral por grupo: 0.5 para hombres, 0.4 para mujeres
umbral <- ifelse(test_data$Sex == "male", 0.5, 0.4)
mitigated_pred <- ifelse(pred_probs > umbral, "good", "bad")

test_data |>
  mutate(orig = pred_class, mitig = mitigated_pred) |>
  group_by(Sex) |>
  summarise(
    fpr_orig = sum(orig == "good" & Risk == "bad") / sum(Risk == "bad"),
    fpr_mitig = sum(mitig == "good" & Risk == "bad") / sum(Risk == "bad"),
    acc_orig = mean(orig == Risk),
    acc_mitig = mean(mitig == Risk),
    .groups = "drop"
  ) |>
  mutate(across(where(is.numeric), ~round(., 3)))

# --- ejercicio-2-auditar-con-otra-variable-proteg --------------

# (código inicial del ejercicio 2; la versión ejecutable está en la solución)
# test_data <- test_data |>
#   mutate(
#     Age_group = case_when(
#       Age < 30 ~ "joven",
#       Age < 50 ~ "adulto",
#       TRUE ~ "mayor"
#     )
#   )
# 
# fobject_age <- fairness_check(
#   explainer,
#   protected = test_data$Age_group,
#   privileged = "adulto",
#   cutoff = 0.5,
#   verbose = FALSE
# )
# 
# plot(fobject_age)

# --- solucion-ejercicio-1 --------------------------------------

codebook_motivos_v2 <- qlm_codebook(
  name = "motivos_no_denuncia_v2",
  instructions = "Leé el fragmento de entrevista. La persona fue víctima de un delito
  y no hizo la denuncia. Clasificá el MOTIVO PRINCIPAL en una de estas categorías:
  - desconfianza: no cree que la policía o la justicia investiguen o resuelvan nada
  - miedo: teme represalias del agresor o de su entorno
  - costo: el trámite implica demasiado tiempo, viajes o burocracia
  - poca_gravedad: considera que el hecho no fue suficientemente grave
  - resolucion_propia: resolvió el problema por su cuenta o por otra vía
  Reglas de decisión, en orden de prioridad:
  1. Si menciona amenazas, represalias o temor por su seguridad o la de su familia,
     es 'miedo', aunque también exprese desconfianza en la policía.
  2. Si el problema ya se resolvió por otra vía (acuerdo, devolución, gestión propia),
     es 'resolucion_propia', aunque el hecho fuera menor.
  3. Si el obstáculo es el trámite (tiempo, distancia, burocracia, sistemas caídos),
     es 'costo', aunque el monto perdido sea chico.
  4. 'poca_gravedad' sólo cuando el argumento central es que el hecho no era serio.
  Elegí UNA sola categoría.",
  schema = type_object(
    motivo = type_enum(
      c("desconfianza", "miedo", "costo", "poca_gravedad", "resolucion_propia"),
      "El motivo principal por el que no denunció"
    ),
    justificacion = type_string("Una oración que justifique la categoría elegida")
  ),
  role = "Sos un investigador experto en criminología y encuestas de victimización.",
  levels = list(motivo = "nominal", justificacion = "nominal")
)

# --- solucion-ejercicio-1-continuacion -------------------------

codificado_v2 <- qlm_code(
  entrevistas$texto,
  codebook_motivos_v2,
  model = "ollama/granite4.1:3b",
  max_active = 1,
  params = params(temperature = 0),
  name = "granite_local_v2"
)

validacion_v2 <- qlm_validate(
  codificado_v2,
  gold = gold,
  by = "motivo",
  level = "nominal"
)

as_tibble(validacion_v2) |> select(measure, value)

# --- solucion-ejercicio-2 --------------------------------------

# Crear grupos de edad
test_data <- test_data |>
  mutate(
    Age_group = case_when(
      Age < 30 ~ "joven",
      Age < 50 ~ "adulto",
      TRUE ~ "mayor"
    )
  )

# Crear objeto fairness por edad
fobject_age <- fairness_check(
  explainer,
  protected = test_data$Age_group,
  privileged = "adulto",
  cutoff = 0.5,
  verbose = FALSE
)

# Visualizar
plot(fobject_age)

# --- apendice-explorar-umbrales-en-detalle ---------------------

# Función para calcular métricas con umbral personalizado
calc_metrics_threshold <- function(data, probs, threshold) {
  pred <- ifelse(probs > threshold, "good", "bad")
  data |>
    mutate(pred = pred) |>
    group_by(Sex) |>
    summarise(
      threshold = threshold,
      accuracy = mean(pred == Risk),
      tpr = sum(pred == "good" & Risk == "good") / sum(Risk == "good"),
      fpr = sum(pred == "good" & Risk == "bad") / sum(Risk == "bad"),
      .groups = "drop"
    )
}

# Probar varios umbrales
thresholds <- c(0.3, 0.4, 0.5, 0.6, 0.7)
results <- map_dfr(thresholds, ~calc_metrics_threshold(test_data, pred_probs, .x))

results |>
  arrange(Sex, threshold) |>
  mutate(across(where(is.numeric), ~round(., 3)))

# --- apendice-visualizar-el-trade-off --------------------------

results |>
  pivot_longer(cols = c(accuracy, tpr, fpr), names_to = "metric", values_to = "value") |>
  ggplot(aes(x = factor(threshold), y = value, color = Sex, group = Sex)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  facet_wrap(~metric, scales = "free_y") +
  scale_color_manual(values = c("female" = "#e63946", "male" = "#2d4563")) +
  labs(title = "Efecto del umbral en métricas por grupo",
       x = "Umbral", y = "Valor") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom")

# --- apendice-cuanta-calidad-cuesta-la-privacidad --------------

# (no se ejecuta por defecto: requiere la API key de OpenRouter)
# codificado_nube <- qlm_replicate(
#   codificado,
#   model = "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
#   max_active = 3,
#   name = "nemotron_nube"
# )
# 
# qlm_compare(
#   codificado, codificado_nube,
#   by = "motivo", level = "nominal"
# )
# 
# qlm_validate(
#   codificado_nube,
#   gold = gold, by = "motivo", level = "nominal"
# )

