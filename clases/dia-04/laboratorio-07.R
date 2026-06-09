# ============================================================
# IA para Científicos Sociales - UCU
# Laboratorio 7: Primeros pasos con ellmer
#
# Autor: Danilo Freire
# Fecha: abril de 2026
#
# Este script contiene todo el código del Laboratorio 7.
# Antes de empezar, configurar la API key de OpenRouter:
#   Sys.setenv(OPENROUTER_API_KEY = "sk-or-...")
# o agregar la línea OPENROUTER_API_KEY=sk-or-... a .Renviron
# (usethis::edit_r_environ() abre el archivo).
#
# Alternativa local: instalar Ollama (https://ollama.com/)
# y cambiar chat_openrouter() por chat_ollama() en cada función.
# ============================================================


# --- Parte 1: Configuración --------------------------------

# Instalar paquetes (solo la primera vez)
# install.packages(c("ellmer", "tidyverse", "jsonlite"))

paquetes <- c("ellmer", "tidyverse", "jsonlite")

for (pkg in paquetes) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Verificar la API key
stopifnot(Sys.getenv("OPENROUTER_API_KEY") != "")

# Modelo por defecto. Cambiar si OpenRouter está saturado.
MODELO <- "meta-llama/llama-3.2-3b-instruct:free"


# --- Primer chat -------------------------------------------

chat <- chat_openrouter(model = MODELO)

respuesta <- chat$chat("Hola, ¿qué podés hacer?")
cat(respuesta, "\n")

# El historial se mantiene dentro del objeto
chat$chat("¿Cuál fue mi primera pregunta?")


# --- Chat con system prompt --------------------------------

analista <- chat_openrouter(
  model = MODELO,
  system_prompt = "Sos un analista político con experiencia en
  América Latina. Respondés de forma breve y académica, en español,
  citando fuentes cuando es posible. Si no estás seguro, lo decís
  explícitamente."
)

analista$chat("¿Cuáles son los principales desafíos democráticos
              de Uruguay en los últimos diez años?")


# --- Parte 2: Clasificación de textos ----------------------

textos <- read_csv("../dia-03/datos/textos_politicos.csv",
                   show_col_types = FALSE)

glimpse(textos)
textos |> count(tema)


# --- Clasificador zero-shot --------------------------------

clasificar_zero <- function(texto) {
  chat <- chat_openrouter(
    model = MODELO,
    system_prompt = "Clasificá el siguiente texto político en
    EXACTAMENTE una de estas categorías:
    economia, educacion, seguridad, salud, medioambiente.

    Reglas estrictas:
    - Responder SOLO con el nombre de la categoría, en minúsculas
    - Sin tildes, sin puntuación, sin explicación
    - Si dudás, elegir la categoría más probable"
  )
  trimws(tolower(chat$chat(texto)))
}

# Probar con un texto
clasificar_zero(textos$texto[1])


# --- Clasificar en batch -----------------------------------

resultados_zero <- textos |>
  slice(1:20) |>
  mutate(tema_llm = map_chr(texto, clasificar_zero, .progress = TRUE))

# Accuracy global
acc_zero <- mean(resultados_zero$tema == resultados_zero$tema_llm)
cat("Accuracy zero-shot:", round(acc_zero * 100, 1), "%\n")

# Matriz de confusión
table(real = resultados_zero$tema, llm = resultados_zero$tema_llm)

# Inspección de errores
resultados_zero |>
  filter(tema != tema_llm) |>
  select(texto, tema, tema_llm)


# --- Clasificador few-shot ---------------------------------

clasificar_few <- function(texto) {
  chat <- chat_openrouter(
    model = MODELO,
    system_prompt = "Clasificá textos políticos en:
    economia, educacion, seguridad, salud, medioambiente.

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

resultados_few <- textos |>
  slice(1:20) |>
  mutate(tema_llm = map_chr(texto, clasificar_few, .progress = TRUE))

acc_few <- mean(resultados_few$tema == resultados_few$tema_llm)
cat("Accuracy few-shot:", round(acc_few * 100, 1), "%\n")


# --- Parte 3: Comparación con LDA del Día 3 ----------------

# Reemplazar 0.72 por el valor real del Laboratorio 6
accuracy_lda <- 0.72

comparacion <- tribble(
  ~metodo,           ~accuracy,    ~tiempo,    ~costo,
  "LDA (Día 3)",     accuracy_lda, "2 seg",    "$0",
  "LLM zero-shot",   acc_zero,     "~2 min",   "<$0.01",
  "LLM few-shot",    acc_few,      "~2 min",   "<$0.01"
)

comparacion


# --- Apéndice 1: Ollama (modelos locales) ------------------

# Requiere haber descargado el modelo:
#   ollama pull llama3.2

# chat_local <- chat_ollama(model = "llama3.2")
# chat_local$chat("Explicame en una oración qué es un LLM.")
#
# clasificar_ollama <- function(texto) {
#   chat <- chat_ollama(
#     model = "llama3.2",
#     system_prompt = "Clasificá en: economia, educacion, seguridad,
#     salud, medioambiente. Responder SOLO con la categoría."
#   )
#   trimws(tolower(chat$chat(texto)))
# }


# --- Apéndice 2: Tres personalidades, misma pregunta -------

# academico <- chat_openrouter(
#   model = MODELO,
#   system_prompt = "Sos un profesor universitario. Respondé de forma
#   formal y académica."
# )
#
# periodista <- chat_openrouter(
#   model = MODELO,
#   system_prompt = "Sos un periodista de un diario nacional.
#   Respondé de forma clara y accesible."
# )
#
# activista <- chat_openrouter(
#   model = MODELO,
#   system_prompt = "Sos un activista social. Respondé con énfasis
#   en la justicia social y los derechos humanos."
# )
#
# pregunta <- "¿Qué opinás sobre la desigualdad en América Latina?"
# academico$chat(pregunta)
# periodista$chat(pregunta)
# activista$chat(pregunta)


# --- Apéndice 3: Few-shot con razonamiento (CoT) -----------

# clasificar_cot <- function(texto) {
#   chat <- chat_openrouter(
#     model = MODELO,
#     system_prompt = "Clasificá textos políticos en: economia,
#     educacion, seguridad, salud, medioambiente.
#
#     Razoná paso a paso y al final escribí en una línea separada:
#     CATEGORIA: <una de las cinco>"
#   )
#   respuesta <- chat$chat(texto)
#   cat_line <- str_extract(respuesta, "CATEGORIA:\\s*\\w+")
#   trimws(tolower(str_remove(cat_line, "CATEGORIA:\\s*")))
# }


# --- Apéndice 4: NER con prompt JSON -----------------------

# ner <- chat_openrouter(
#   model = MODELO,
#   system_prompt = 'Extraé entidades nombradas del texto.
#   Respondé en JSON estricto con esta estructura:
#   {
#     "personas": [],
#     "organizaciones": [],
#     "lugares": []
#   }
#   NO incluir texto fuera del JSON.'
# )
#
# respuesta <- ner$chat(textos$texto[5])
# cat(respuesta)
# fromJSON(respuesta)


# ============================================================
# Fin del Laboratorio 7. Próximo: laboratorio-08.R
# (extracción estructurada, datos sintéticos, auditoría de sesgos)
# ============================================================
