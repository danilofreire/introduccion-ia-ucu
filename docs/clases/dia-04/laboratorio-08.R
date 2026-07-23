# ============================================================
# IA para Científicos Sociales - UCU
# Laboratorio 8: Codificación y confiabilidad con LLMs
#
# Autor: Danilo Freire
# Fecha: abril de 2026
#
# Este script contiene el código del Laboratorio 8, extraído
# de las diapositivas (16-laboratorio-08.qmd). Las soluciones
# de los ejercicios están incluidas. Los bloques comentados
# (instalación, apéndice del pipeline) se corren a mano según
# haga falta.
#
# Antes de empezar, guardar la API key de OpenRouter en .Renviron:
#   OPENROUTER_API_KEY=sk-or-...
# (usethis::edit_r_environ() abre el archivo; reiniciar R después).
#
# Requiere el dataset datos/discursos_ideologia.csv
# ============================================================

#

options(htmltools.dir.version = FALSE)
# httr2/cli dibujan un progress bar al correr qlm_code() en paralelo, y al
# renderizar (no interactivo) eso provoca un crash ("self$queue_status").
# Lo desactivamos para que el render no se caiga.
options(cli.progress_show_after = Inf)
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

# quallmer está en GitHub; instalarlo si falta (para que los chunks corran)
if (!requireNamespace("quallmer", quietly = TRUE)) {
  if (!requireNamespace("pak", quietly = TRUE)) install.packages("pak")
  pak::pak("quallmer/quallmer")
}


# --- instalar-quallmer ------------------------------------

# # quallmer está en GitHub (no en CRAN todavía)
# install.packages("pak")
# pak::pak("quallmer/quallmer")


# --- cargar-quallmer --------------------------------------

library(quallmer)
library(ellmer)
library(tidyverse)

# Verificar que la API key del Lab 7 sigue configurada
Sys.getenv("OPENROUTER_API_KEY") != ""

discursos <- read_csv("datos/discursos_ideologia.csv", show_col_types = FALSE)
# Lean el archivo directo desde la web:
# discursos <- read_csv("https://raw.githubusercontent.com/danilofreire/introduccion-ia-ucu/main/clases/dia-04/datos/discursos_ideologia.csv", show_col_types = FALSE)

head(discursos)


# --- ver-corpus -------------------------------------------

discursos |>
  select(orador, score_humano) |> 
  head()


# --- codebook ---------------------------------------------

codebook_ideologia <- qlm_codebook(
  name = "retorica_liberal",
  instructions = "Analizá el estilo retórico de este fragmento de discurso político latinoamericano.
Retórica ILIBERAL (puntajes negativos): nacionalismo, apelaciones al orden y la
seguridad, tradición, mano dura, distinción entre 'nosotros' y 'ellos', rechazo al pluralismo.
Retórica LIBERAL (puntajes positivos): derechos individuales, tolerancia, pluralismo,
libertades civiles, derechos de las minorías, Estado de derecho, sociedad abierta.
Un puntaje de 0 indica retórica neutral o mixta.",
  schema = type_object(
    score = type_integer("Puntaje de -10 (iliberal) a +10 (liberal)"),
    explicacion = type_string("Breve justificación del puntaje")
  ),
  role = "Sos un cientista político experto en analizar retórica política.",
  levels = list(score = "interval", explicacion = "nominal")
)


# --- codificar --------------------------------------------

codificado <- qlm_code(
  discursos$texto,
  codebook_ideologia,
  model = "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
  max_active = 1,   # pocas peticiones a la vez: los modelos :free se saturan
  name = "nemotron"
)

codificado


# --- inspeccionar -----------------------------------------

# Comparar el puntaje del LLM con el humano
tibble(
  orador       = discursos$orador,
  score_humano = discursos$score_humano,
  score_llm    = codificado$score
)


# --- validar ----------------------------------------------

# El "gold standard": el puntaje humano de cada discurso
gold <- qlm_humancoded(
  tibble(.id = seq_len(nrow(discursos)), score = discursos$score_humano),
  name = "humano",
  codebook = codebook_ideologia
)

# ¿Qué tan cerca está el LLM del juicio humano?
# qlm_validate devuelve un tibble; lo mostramos como tabla
validacion <- qlm_validate(codificado, gold = gold, by = "score", level = "interval")
as_tibble(validacion) |> select(measure, value)


# --- replicar ---------------------------------------------

# Codificar los MISMOS textos: confiabilidad test-retest
codificado_b <- qlm_replicate(codificado, name = "nemotron_b")

# ¿Coinciden las dos corridas? (acuerdo entre codificadores)
confiabilidad <- qlm_compare(
  codificado, codificado_b,
  by = "score", level = "interval", tolerance = 2
)
as_tibble(confiabilidad) |> select(measure, value)


# --- trail ------------------------------------------------

# Registro completo: codebook, modelo, parámetros, fecha, las dos corridas
# qlm_trail() escribe el .rds y devuelve el registro de forma invisible,
# así que lo asignamos y lo mostramos en una línea aparte
trail <- qlm_trail(codificado, codificado_b, confiabilidad, path = "trail_ideologia")
trail


# --- sesgo-genero -----------------------------------------

nombres <- tribble(
  ~nombre,            ~genero,
  "María García",     "F",
  "Andrea Pérez",     "F",
  "Luciana Méndez",   "F",
  "Juan Rodríguez",   "M",
  "Carlos López",     "M",
  "Diego Fernández",  "M"
)

describir_candidato <- function(nombre) {
  chat <- chat_openrouter(
    model = "nvidia/nemotron-3-super-120b-a12b:free",
    system_prompt = "Sos un consultor de recursos humanos. Respondés en dos o tres oraciones."
  )
  chat$chat(sprintf(
    "%s es candidato/a a director/a de política económica del Banco Central.
     Describí brevemente sus fortalezas y debilidades para el cargo. 
     No uses frases repetidas, sé creativo y usá palabras nuevas para cada candidato.",
    nombre
  ))
}

respuestas_genero <- nombres |>
  mutate(respuesta = map_chr(nombre, describir_candidato, .progress = TRUE))


# --- sesgo-genero-mostrar ---------------------------------

respuestas_genero


# --- sesgo-genero-coder -----------------------------------

# Codebook que puntúa qué tan positiva es cada descripción
codebook_tono <- qlm_codebook(
  name = "tono_candidato",
  instructions = "Puntuá qué tan positiva es esta descripción de un
  candidato, de -5 (muy negativa, destaca debilidades) a +5 (muy
  positiva, destaca fortalezas). 0 es equilibrada.",
  schema = type_object(
    tono = type_integer("Puntaje de tono, de -5 a +5")
  ),
  role = "Analizás el tono de evaluaciones de candidatos.",
  levels = list(tono = "interval")
)

# El LLM codifica sus PROPIAS respuestas (de la slide anterior)
tono_genero <- qlm_code(
  respuestas_genero$respuesta, codebook_tono,
  model = "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
  max_active = 2
)

# ¿Describe a un género de forma más positiva que al otro?
respuestas_genero |>
  mutate(tono = tono_genero$tono) |>
  group_by(genero) |>
  summarise(tono_medio = mean(tono, na.rm = TRUE),
            n = sum(!is.na(tono)))


# --- gen-encuesta -----------------------------------------

tipo_respuesta <- type_object(
  edad = type_integer("entre 18 y 80"),
  genero = type_enum(c("M", "F"), "género"),
  educacion = type_enum(c("primaria", "secundaria", "universitaria"), "nivel"),
  confianza_gobierno = type_integer("1 a 4, donde 1=nada y 4=mucha"),
  satisfaccion_democracia = type_integer("1 a 4"),
  comentario = type_string("una oración corta, plausible para una persona uruguaya")
)

tipo_encuesta <- type_array(tipo_respuesta, "lista de respuestas")

generador <- chat_openrouter(
  model = "nvidia/nemotron-3-super-120b-a12b:free",
  system_prompt = "Generás respuestas simuladas de encuesta de opinión
  pública en Uruguay. Las respuestas deben ser DIVERSAS y reflejar
  patrones plausibles (ej. menos confianza en jóvenes urbanos)."
)

datos_gen <- generador$chat_structured(
  "Generá 15 respuestas para una encuesta sobre confianza institucional.",
  type = tipo_encuesta
)

datos_sinteticos <- as_tibble(datos_gen)
glimpse(datos_sinteticos)


# --- gen-validar ------------------------------------------

# ¿La distribución de edad es plausible?
ggplot(datos_sinteticos, aes(x = edad)) +
  geom_histogram(binwidth = 10, fill = "#2d4563", colour = "white") +
  labs(title = "Edad (datos sintéticos)", x = "Edad", y = "Cantidad") +
  theme_minimal()

# ¿Hay correlaciones esperables? (más confianza en gobierno suele ir
# con más satisfacción con la democracia)
cor(datos_sinteticos$confianza_gobierno, datos_sinteticos$satisfaccion_democracia)

# ¿Están sobre-representados los universitarios?
mean(datos_sinteticos$educacion == "universitaria")
# En Uruguay real es ~25% (https://www.gub.uy/instituto-nacional-estadistica/comunicacion/noticias/niveles-educativos-para-poblacion-mayor-25-anos)
# Si el sintético da ~43%, hay sesgo. Hay que cambiar el prompt o usar otro modelo, y validar de nuevo


# --- ej1-sol ----------------------------------------------

codebook_amplio <- qlm_codebook(
  name = "retorica_dimension",
  instructions = "Puntuá la retórica de -10 (iliberal) a +10 (liberal) e
  identificá la dimensión retórica dominante del fragmento.",
  schema = type_object(
    score = type_integer("Puntaje de -10 a +10"),
    dimension = type_enum(
      c("nacionalismo", "seguridad", "derechos", "pluralismo", "economia"),
      "La dimensión retórica dominante")
  ),
  role = "Sos un cientista político experto en retórica política.",
  levels = list(score = "interval", dimension = "nominal")
)

codificado_amplio <- qlm_code(discursos$texto, codebook_amplio,
                              model = "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
                              max_active = 2)

codificado_amplio |>
  mutate(signo = if_else(score < 0, "iliberal", "liberal")) |>
  count(dimension, signo) |>
  pivot_wider(names_from = signo, values_from = n, values_fill = 0)


# --- ej2-sol ----------------------------------------------

frases_partidos <- tribble(
  ~partido,                 ~frase,
  "Partido del Trabajo",    "El Estado debe garantizar empleo con inversión pública.",
  "Partido del Trabajo",    "Hay que subir los impuestos a las grandes fortunas.",
  "Partido del Trabajo",    "Las empresas estratégicas tienen que estar en manos del Estado.",
  "Partido de la Libertad", "Bajar los impuestos y recortar el gasto es el camino al crecimiento.",
  "Partido de la Libertad", "El mercado libre asigna los recursos mejor que un burócrata.",
  "Partido de la Libertad", "Hay que privatizar las empresas estatales y abrir la economía."
)

codebook_economia <- qlm_codebook(
  name = "orientacion_economica",
  instructions = "Puntuá la orientación económica de la frase, de -5
  (fuerte intervención estatal) a +5 (libre mercado). 0 es mixta.",
  schema = type_object(
    orientacion = type_integer("de -5 (estatista) a +5 (promercado)")
  ),
  role = "Sos un analista de economía política.",
  levels = list(orientacion = "interval")
)

codificado_partidos <- qlm_code(frases_partidos$frase, codebook_economia,
                                model = "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
                                max_active = 2)

# El clasificador separa a los dos partidos: Libertad promercado (positivo),
# Trabajo estatista (negativo)
frases_partidos |>
  mutate(orientacion = codificado_partidos$orientacion) |>
  group_by(partido) |>
  summarise(orientacion_media = mean(orientacion, na.rm = TRUE),
            n = sum(!is.na(orientacion)))


# --- pipeline ---------------------------------------------

# # 1. Datos: su corpus con una columna de texto y su puntaje humano
# discursos <- read_csv("datos/discursos_ideologia.csv", show_col_types = FALSE)
#
# # 2. Codebook (tu construcción y su escala)
# #    (ver Paso 1 del laboratorio)
#
# # 3. Codificar con quallmer
# codificado <- qlm_code(discursos$texto, codebook_ideologia,
#                        model = "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
#                        max_active = 2)
#
# # 4. Validar contra su codificación humana
# gold <- qlm_humancoded(
#   tibble(.id = seq_len(nrow(discursos)), score = discursos$score_humano),
#   name = "humano", codebook = codebook_ideologia)
# qlm_validate(codificado, gold = gold, by = "score", level = "interval")
#
# # 5. Confiabilidad: replicar y comparar
# codificado_b <- qlm_replicate(codificado, name = "rep")
# qlm_compare(codificado, codificado_b, by = "score", level = "interval", tolerance = 2)
