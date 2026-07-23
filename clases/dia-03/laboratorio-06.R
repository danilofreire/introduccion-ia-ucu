# ============================================================
# IA para Científicos Sociales - UCU
# Laboratorio 6: Análisis de texto
#
# Autor: Danilo Freire
# Fecha: mayo de 2026
#
# Este script contiene todo el código del Laboratorio 6.
# Cada sección coincide con una diapositiva (DEMO o EJERCICIO).
# Las soluciones de los ejercicios están incluidas tal como
# aparecen en los apéndices del .qmd.
# ============================================================


# --- Cargar paquetes -----------------------------------------

# Si algún paquete falta:
# install.packages(c("tidyverse", "tidytext", "tm", "ggwordcloud"))

library(tidyverse)
library(tidytext)      # análisis de texto tidy
library(tm)            # stopwords en español
library(ggwordcloud)   # nubes de palabras (apéndice)

set.seed(2026)


# --- Parte 1: Tokenización y limpieza ------------------------

# Cargar el corpus simulado (60 textos políticos)
textos <- read_csv("datos/textos_politicos.csv", show_col_types = FALSE)
# Lean el archivo directo desde la web:
# textos <- read_csv("https://raw.githubusercontent.com/danilofreire/introduccion-ia-ucu/main/clases/dia-03/datos/textos_politicos.csv", show_col_types = FALSE)

glimpse(textos)


# --- Distribución de textos por tema -------------------------

textos |>
  count(tema)


# --- Ver un texto de ejemplo ---------------------------------

# Un texto de cada tema
textos |>
  group_by(tema) |>
  slice(1) |>
  select(tema, texto) |>
  ungroup()


# --- Tokenizar -----------------------------------------------

# Una fila por palabra; unnest_tokens() convierte a minúsculas
# y elimina puntuación automáticamente
tokens <- textos |>
  unnest_tokens(palabra, texto)

tokens |> head(20)

# Para bigramas:
# unnest_tokens(palabra, texto, token = "ngrams", n = 2)


# --- Stopwords en español ------------------------------------

stopwords_es <- tibble(palabra = tm::stopwords("spanish"))

head(stopwords_es, 20)
cat("Número de stopwords:", nrow(stopwords_es), "\n")


# --- Eliminar stopwords --------------------------------------

# Quitar stopwords, números y palabras de menos de 3 letras
tokens_limpios <- tokens |>
  anti_join(stopwords_es, by = "palabra") |>
  filter(!str_detect(palabra, "^[0-9]+$"),
         nchar(palabra) > 2)

cat("Tokens antes:", nrow(tokens), "\n")
cat("Tokens después:", nrow(tokens_limpios), "\n")
cat("Reducción:", round((1 - nrow(tokens_limpios) / nrow(tokens)) * 100, 1), "%\n")


# --- Palabras más frecuentes ---------------------------------

tokens_limpios |>
  count(palabra, sort = TRUE) |>
  head(20) |>
  mutate(palabra = reorder(palabra, n)) |>
  ggplot(aes(x = n, y = palabra)) +
  geom_col(fill = "#2d4563") +
  labs(title = "20 palabras más frecuentes (sin stopwords)",
       x = "Frecuencia", y = NULL) +
  theme_minimal()


# --- Parte 2: TF-IDF -----------------------------------------

# Contar palabras por tema
palabras_tema <- tokens_limpios |>
  count(tema, palabra, sort = TRUE)

# Calcular TF-IDF (palabra, grupo, frecuencia)
tfidf <- palabras_tema |>
  bind_tf_idf(palabra, tema, n)

# Top 15 palabras más distintivas a nivel global
tfidf |>
  arrange(desc(tf_idf)) |>
  head(15)


# --- Visualizar TF-IDF por tema ------------------------------

tfidf |>
  group_by(tema) |>
  slice_max(tf_idf, n = 6, with_ties = FALSE) |>
  ungroup() |>
  mutate(palabra = reorder_within(palabra, tf_idf, tema)) |>
  ggplot(aes(x = tf_idf, y = palabra, fill = tema)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~tema, scales = "free_y", ncol = 3) +
  scale_y_reordered() +
  labs(title = "Palabras más distintivas por tema (TF-IDF)",
       x = "TF-IDF", y = NULL) +
  theme_minimal()


# --- Ejercicio rápido: TF-IDF por país (solución) ------------

tfidf_pais <- tokens_limpios |>
  count(pais, palabra) |>
  bind_tf_idf(palabra, pais, n)

# Top 5 palabras distintivas por país
tfidf_pais |>
  group_by(pais) |>
  slice_max(tf_idf, n = 5, with_ties = FALSE) |>
  arrange(pais, desc(tf_idf))

# Para ver TODAS las palabras: sin slice_max() y con print(n = Inf), porque
# las tibbles muestran solo diez filas por defecto (o View(tfidf_pais))
tfidf_pais |> arrange(pais, desc(tf_idf)) |> print(n = Inf)

# Lecciones:
# - Aparecen palabras locales (instituciones, ciudades) y términos específicos
# - Los TF-IDF más altos suelen ser palabras únicas a un país


# --- Parte 3: Análisis de sentimiento ------------------------

# Diccionarios para texto político (versión inicial — ampliable)
positivas <- c("crecimiento", "mejora", "mejorado", "oportunidad",
               "fortalecido", "innovadores", "renovables", "sólido",
               "beneficio", "avance", "estabilizar", "reducir")

negativas <- c("crisis", "inflación", "pobreza", "violencia", "amenaza",
               "problema", "deuda", "riesgo", "deficiencias", "grave",
               "crimen", "narcotráfico", "deforestación", "informalidad",
               "inseguridad", "contaminación", "desempleo", "vulnerables")

cat("Positivas:", length(positivas), "  Negativas:", length(negativas), "\n")


# --- Score por documento -------------------------------------

# +1 para positivas, -1 para negativas, 0 para las demás
sentimiento_doc <- tokens_limpios |>
  mutate(valencia = case_when(
    palabra %in% positivas ~  1L,
    palabra %in% negativas ~ -1L,
    TRUE ~ 0L
  )) |>
  group_by(id, tema, pais) |>
  summarise(score = sum(valencia), .groups = "drop")

# Documentos más positivos
sentimiento_doc |>
  arrange(desc(score)) |>
  head(3)

# Documentos más negativos
sentimiento_doc |>
  arrange(score) |>
  head(3)


# --- Visualizar tono por tema --------------------------------

sentimiento_doc |>
  group_by(tema) |>
  summarise(score_promedio = mean(score), .groups = "drop") |>
  mutate(tema = reorder(tema, score_promedio)) |>
  ggplot(aes(x = score_promedio, y = tema, fill = score_promedio > 0)) +
  geom_col(show.legend = FALSE) +
  scale_fill_manual(values = c("TRUE" = "#27AE60", "FALSE" = "#E74C3C")) +
  labs(title = "Tono promedio por tema",
       x = "Score promedio (positivas − negativas)", y = NULL) +
  theme_minimal()


# --- Ejercicio rápido: doc más negativo (solución) -----------

# Doc con score más negativo
doc_mas_negativo <- sentimiento_doc |>
  arrange(score) |>
  head(1)

doc_mas_negativo

# Recuperar el texto completo
textos |>
  filter(id == doc_mas_negativo$id) |>
  pull(texto)

# Reflexión:
# - El método captura bien temas con palabras negativas explícitas
#   (seguridad, crisis, violencia)
# - Cuando el tono está implícito (ironía, comparaciones) lo pierde
# - Combinar siempre con lectura humana de una muestra


# --- Parte 4: Síntesis ---------------------------------------

# TF-IDF responde: ¿de qué habla el texto?
# Sentimiento responde: ¿cómo lo dice?
#
# Combinadas, construyen una narrativa:
# "Los textos sobre seguridad (TF-IDF) tienen tono marcadamente
# negativo (sentimiento)."


# ============================================================
# MATERIAL OPCIONAL (apéndices del laboratorio)
# ============================================================


# --- Apéndice 2: Nube de palabras ----------------------------

freq <- tokens_limpios |>
  count(palabra, sort = TRUE) |>
  head(100)

ggplot(freq, aes(label = palabra, size = n, color = n)) +
  geom_text_wordcloud(rm_outside = TRUE) +
  scale_size_area(max_size = 15) +
  scale_color_gradient(low = "#457b9d", high = "#e63946") +
  theme_minimal() +
  labs(title = "Nube de palabras del corpus")
