# ============================================================
# IA para Científicos Sociales - UCU
# Laboratorio 7: Primeros pasos con ellmer
#
# Autor: Danilo Freire
# Fecha: abril de 2026
#
# Este script contiene el código del Laboratorio 7, extraído
# de las diapositivas (15-laboratorio-07.qmd). Las soluciones
# de los ejercicios están incluidas. Los bloques comentados
# (instalación, configuración de la key, apéndices) se corren
# a mano según haga falta.
#
# Antes de empezar, guardar la API key de OpenRouter en .Renviron:
#   OPENROUTER_API_KEY=sk-or-...
# (usethis::edit_r_environ() abre el archivo; reiniciar R después).
# ============================================================

#

options(htmltools.dir.version = FALSE)
library(knitr)
opts_chunk$set(
  prompt = T,
  fig.align = "center",
  dpi = 300,
  cache = T,
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

if (!require("fontawesome", character.only = TRUE)) {
  install.packages("fontawesome", dependencies = TRUE)
  library(fontawesome, character.only = TRUE)
}


# --- instalar-ellmer -----------------------------------------

# Instalar (solo la primera vez)
# install.packages(c("ellmer", "jsonlite"))


# --- cargar-paquetes -----------------------------------------

library(tidyverse)
library(ellmer)
library(jsonlite)

packageVersion("ellmer")


# --- configurar-api ------------------------------------------

# Opción 1: solo para esta sesión de R
# Sys.setenv(OPENROUTER_API_KEY = "sk-or-...")

# Opción 2 (recomendada): guardar en .Renviron permanentemente
# Editar el archivo con:
# usethis::edit_r_environ()
# Agregar la línea (sin comillas):
# OPENROUTER_API_KEY=sk-or-...
# Guardar y reiniciar R

# Verificar
# Sys.getenv("OPENROUTER_API_KEY") != ""   # TRUE si está bien


# --- primer-chat ---------------------------------------------

chat <- chat_openrouter(model = "openai/gpt-oss-20b:free")

# Mensaje simple
respuesta <- chat$chat("Hola, ¿qué podés hacer?")

# El historial se mantiene dentro del objeto chat
chat$chat("¿Cuál fue mi primera pregunta?")


# --- system-prompt -------------------------------------------

analista <- chat_openrouter(
  model = "openai/gpt-oss-20b:free",
  system_prompt = "Sos un analista político con experiencia en América Latina.
  Respondés de forma breve y académica, en español,
  citando fuentes cuando es posible. Si no estás seguro, lo decís explícitamente."
)

analista$chat("¿Cuáles son los principales desafíos democráticos de Uruguay en los últimos diez años?")


# --- cargar-textos -------------------------------------------

textos <- read_csv("datos/textos_politicos.csv",
                   show_col_types = FALSE)
# Lean el archivo directo desde la web:
# textos <- read_csv("https://raw.githubusercontent.com/danilofreire/introduccion-ia-ucu/main/clases/dia-04/datos/textos_politicos.csv", show_col_types = FALSE)

glimpse(textos)

# El dataset está ordenado por tema: armamos una muestra balanceada
set.seed(2026)
muestra <- textos |>
  group_by(tema) |>
  slice_sample(n = 4) |>     # 4 por tema = 20 textos, las 5 categorías
  ungroup() |>
  slice_sample(prop = 1)     # mezclar el orden

muestra |> count(tema)


# --- clasificador-zero ---------------------------------------

clasificar_zero <- function(texto) {
  chat <- chat_openrouter(
    model = "openai/gpt-oss-20b:free",
    system_prompt = "Clasificá el siguiente texto político en EXACTAMENTE una de estas categorías:
    economia, educacion, seguridad, salud, medioambiente.

    Reglas estrictas:
    - Responder SOLO con el nombre de la categoría, en minúsculas
    - Sin tildes, sin puntuación, sin explicación
    - Si dudás, elegir la categoría más probable"
  )
  trimws(tolower(chat$chat(texto)))
}

# Probar con un texto
muestra$texto[1]
clasificar_zero(muestra$texto[1])


# --- clasificar-batch ----------------------------------------

# Clasificar los 20 textos de la muestra balanceada
resultados_zero <- muestra |>
  mutate(tema_llm = map_chr(texto, clasificar_zero, .progress = TRUE))


# --- clasificar-batch-mostrar --------------------------------

resultados_zero |>
  select(tema, tema_llm, texto) |>
  head(8)


# --- evaluar-zero --------------------------------------------

evaluacion <- resultados_zero |>
  mutate(correcto = tema == tema_llm)

# Accuracy global
mean(evaluacion$correcto)

# Matriz de confusión
table(real = evaluacion$tema, llm = evaluacion$tema_llm)

# ¿Dónde se equivoca más?
evaluacion |>
  filter(!correcto) |>
  count(tema, tema_llm, sort = TRUE)


# --- clasificador-few ----------------------------------------

clasificar_few <- function(texto) {
  chat <- chat_openrouter(
    model = "openai/gpt-oss-20b:free",
    system_prompt = "Clasificá textos políticos en: economia, educacion, seguridad, salud, medioambiente.

    Ejemplos:
    - 'La inflación volvió a subir este mes' -> economia
    - 'Los docentes piden mejores salarios' -> educacion
    - 'Aumentaron los robos en la capital' -> seguridad
    - 'Se inauguró un nuevo hospital' -> salud
    - 'Los incendios forestales se extienden' -> medioambiente

    Responder SOLO con la categoría, sin tildes ni explicación."
  )
  trimws(tolower(chat$chat(texto)))
}

resultados_few <- muestra |>
  mutate(tema_llm = map_chr(texto, clasificar_few, .progress = TRUE))


# --- clasificador-few-mostrar --------------------------------

mean(resultados_few$tema == resultados_few$tema_llm)


# --- comparar-modelos ----------------------------------------

# La misma tarea con dos modelos: ¿cambian velocidad y resultados?
clasificar_con <- function(texto, modelo) {
  chat <- chat_openrouter(
    model = modelo,
    system_prompt = "Clasificá en: economia, educacion, seguridad, salud, medioambiente. Responder SOLO con la categoría, en minúsculas y sin tildes."
  )
  trimws(tolower(chat$chat(texto)))
}

prueba <- muestra |> slice(1:8)

# Un modelo rápido vs uno más cuidadoso
prueba <- prueba |>
  mutate(
    gpt_oss  = map_chr(texto, clasificar_con, "openai/gpt-oss-20b:free"),
    nemotron = map_chr(texto, clasificar_con, "nvidia/nemotron-nano-9b-v2:free")
  )


# --- comparar-modelos-mostrar --------------------------------

# ¿Coinciden entre sí los dos modelos?
mean(prueba$gpt_oss == prueba$nemotron)

# Guardar los resultados para reusarlos después (CSV)
write_csv(prueba, "clasificacion_lab7.csv")


# --- lda-corpus ----------------------------------------------

library(tidytext)
library(topicmodels)
# Si falta algún paquete: install.packages(c("tidytext", "topicmodels", "tm"))

# 1. Tokenizar y quitar stopwords (igual que el Día 3)
extra_stop <- c("sigue", "siendo", "sido", "ser", "hacer", "ha", "han",
  "hay", "más", "país", "países", "región", "regiones", "gobierno",
  "toda", "todos", "todas")
stop_es <- tibble(palabra = unique(c(tm::stopwords("spanish"), extra_stop)))

tokens_limpios <- textos |>
  select(id, tema, texto) |>
  unnest_tokens(palabra, texto) |>
  filter(!str_detect(palabra, "^[0-9]+$")) |>
  anti_join(stop_es, by = "palabra")

# 2. Matriz documento-término y LDA con k = 5
dtm <- tokens_limpios |> count(id, palabra) |> cast_dtm(id, palabra, n)
set.seed(123)
modelo_lda <- LDA(dtm, k = 5, control = list(seed = 123))

# 3. Tópico dominante de cada documento (gamma)
doc_topic <- tidy(modelo_lda, matrix = "gamma") |>
  group_by(document) |>
  slice_max(gamma, n = 1, with_ties = FALSE) |>
  ungroup() |>
  transmute(id = as.integer(document), topic)

# 4. Mapear cada tópico al tema real más frecuente (para medir accuracy)
mapa <- doc_topic |>
  left_join(select(textos, id, tema), by = "id") |>
  count(topic, tema) |>
  group_by(topic) |>
  slice_max(n, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(topic, tema_lda = tema)

lda_pred <- doc_topic |>
  left_join(mapa, by = "topic") |>
  left_join(select(textos, id, tema), by = "id")

# Medir solo sobre los 20 textos de la muestra (mismo set que los LLMs)
accuracy_lda <- lda_pred |>
  filter(id %in% muestra$id) |>
  summarise(acc = mean(tema == tema_lda)) |>
  pull(acc)
accuracy_lda


# --- clasif-gptoss-corpus ------------------------------------

# resultados_zero (Parte 2) = gpt-oss-20b zero-shot sobre la muestra
accuracy_gptoss <- mean(resultados_zero$tema == resultados_zero$tema_llm)
accuracy_gptoss


# --- clasif-nemotron-corpus ----------------------------------

clasif_nemotron <- muestra |>
  mutate(tema_llm = map_chr(texto, clasificar_con, "nvidia/nemotron-nano-9b-v2:free", .progress = TRUE))


# --- clasif-nemotron-corpus-mostrar --------------------------

accuracy_nemotron <- mean(clasif_nemotron$tema == clasif_nemotron$tema_llm)
accuracy_nemotron


# --- comparar ------------------------------------------------

comparacion <- tribble(
  ~metodo,                 ~accuracy,          ~tiempo,         ~costo,
  "LDA (no supervisado)",  accuracy_lda,       "~2 seg",        "$0",
  "gpt-oss-20b",           accuracy_gptoss,    "ya en Parte 2", "<$0.01",
  "nemotron-nano-9b",      accuracy_nemotron,  "~30 seg",       "<$0.01"
)

comparacion |> arrange(desc(accuracy))


# --- ej1-sol -------------------------------------------------

academico <- chat_openrouter(
  model = "openai/gpt-oss-20b:free",
  system_prompt = "Sos un profesor universitario. Respondé de forma formal y académica, con referencias a literatura."
)
periodista <- chat_openrouter(
  model = "openai/gpt-oss-20b:free",
  system_prompt = "Sos un periodista de un diario nacional. Respondé de forma clara y accesible para el público general."
)
activista <- chat_openrouter(
  model = "openai/gpt-oss-20b:free",
  system_prompt = "Sos un activista social. Respondé con énfasis en la justicia social y los derechos humanos."
)

pregunta <- "¿Qué opinás sobre la desigualdad en América Latina?"
academico$chat(pregunta)
periodista$chat(pregunta)
activista$chat(pregunta)


# --- ej2-sol -------------------------------------------------

parrafo1 <- "El gobierno anunció un paquete de medidas económicas que combina
recortes en el gasto público con incentivos a la inversión privada. La oposición
sostiene que el ajuste recaerá sobre los más vulnerables, mientras que los
empresarios celebran la previsibilidad fiscal."

parrafo2 <- "La reforma educativa propone extender la jornada escolar y actualizar
los contenidos en ciencia y tecnología. Los sindicatos docentes reclaman que
llega sin presupuesto para infraestructura ni para mejorar los salarios. Varias
universidades ofrecieron colaborar en la formación de profesores."

resumir <- function(texto) {
  chat <- chat_openrouter(
    model = "openai/gpt-oss-20b:free",
    system_prompt = "Resumí el texto en UNA sola oración bien corta, en español, sin opinar."
  )
  chat$chat(texto)
}

resumir(parrafo1)
resumir(parrafo2)


# --- analizar-errores ----------------------------------------

errores <- evaluacion |>
  filter(!correcto) |>
  select(texto, tema, tema_llm)

errores

# ¿Son errores "razonables"? A veces el LLM elige un tema
# distinto pero igual de defendible (ej. salud vs medioambiente
# en un texto sobre contaminación que afecta la salud)


# --- cot -----------------------------------------------------

clasificar_cot <- function(texto) {
  chat <- chat_openrouter(
    model = "openai/gpt-oss-20b:free",
    system_prompt = "Clasificá textos políticos en: economia,
    educacion, seguridad, salud, medioambiente.

    Razoná paso a paso (qué palabras clave aparecen, qué institución
    se menciona, qué política se debate) y al final escribí en una
    línea separada:
    CATEGORIA: <una de las cinco>"
  )

  respuesta <- chat$chat(texto)
  # Extraer la línea final
  cat_line <- str_extract(respuesta, "CATEGORIA:\\s*\\w+")
  trimws(tolower(str_remove(cat_line, "CATEGORIA:\\s*")))
}

clasificar_cot(textos$texto[1])


# --- ner -----------------------------------------------------

ner <- chat_openrouter(
  model = "openai/gpt-oss-20b:free",
  system_prompt = 'Extraé entidades nombradas del texto.
  Respondé en JSON estricto con esta estructura:
  {
    "personas": [],
    "organizaciones": [],
    "lugares": []
  }
  NO incluir texto fuera del JSON.'
)

respuesta <- ner$chat(textos$texto[5])
cat(respuesta)

# Parsear
fromJSON(respuesta)


