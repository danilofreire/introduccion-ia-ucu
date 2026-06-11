# Script para crear el dataset de la Tarea 4
# Corte transversal de 400 escuelas (datos simulados, inspirados en pruebas
# estandarizadas tipo Aristas/PISA)
#
# Diseño:
#   - El nivel socioeconómico y la asistencia son los motores principales del
#     puntaje; experiencia docente, tamaño de clase y presupuesto tienen
#     efectos moderados.
#   - computadoras_alumno correlaciona fuerte con presupuesto_alumno pero NO
#     tiene efecto directo, y horas_semanales no tiene efecto: candidatas
#     naturales a que LASSO las elimine.

library(tibble)

set.seed(2026)

departamentos <- c("Montevideo", "Canelones", "Maldonado", "Salto",
                   "Paysandú", "Rivera", "Tacuarembó", "Colonia")

n <- 400

escuelas <- tibble(
  escuela = sprintf("escuela_%03d", 1:n),
  departamento = sample(departamentos, n, replace = TRUE),
  sector = sample(c("publico", "privado"), n, replace = TRUE, prob = c(0.72, 0.28)),
  zona = sample(c("urbana", "rural"), n, replace = TRUE, prob = c(0.68, 0.32))
)

# Nivel socioeconómico: más alto en privadas y urbanas
escuelas$nivel_socioeconomico <- round(pmin(100, pmax(5,
  48 +
  14 * (escuelas$sector == "privado") +
  7 * (escuelas$zona == "urbana") +
  rnorm(n, 0, 14))), 1)

escuelas$asistencia_pct <- round(pmin(100, pmax(50,
  86 + 0.10 * (escuelas$nivel_socioeconomico - 48) + rnorm(n, 0, 5))), 1)

escuelas$experiencia_docente <- round(pmax(1, rnorm(n, 14, 6)), 1)

escuelas$tamano_clase <- round(pmax(8, pmin(45,
  27 - 4 * (escuelas$sector == "privado") + rnorm(n, 0, 5))))

escuelas$presupuesto_alumno <- round(pmax(300,
  1100 +
  9 * (escuelas$nivel_socioeconomico - 48) +
  350 * (escuelas$sector == "privado") +
  rnorm(n, 0, 220)))

# Correlacionada con presupuesto, SIN efecto directo sobre el puntaje
escuelas$computadoras_alumno <- round(pmax(0,
  0.15 + 0.00035 * escuelas$presupuesto_alumno + rnorm(n, 0, 0.12)), 2)

# Sin efecto sobre el puntaje
escuelas$horas_semanales <- round(pmax(20, pmin(45, rnorm(n, 30, 4))), 1)

escuelas$formacion_docente_pct <- round(pmin(100, pmax(10,
  55 + rnorm(n, 0, 18))), 1)

# Outcome: puntaje en una prueba estandarizada (media ~500)
escuelas$puntaje_prueba <- round(
  500 +
  1.6 * (escuelas$nivel_socioeconomico - 48) +
  2.2 * (escuelas$asistencia_pct - 86) +
  1.1 * (escuelas$experiencia_docente - 14) -
  1.3 * (escuelas$tamano_clase - 27) +
  0.025 * (escuelas$presupuesto_alumno - 1100) +
  0.15 * (escuelas$formacion_docente_pct - 55) +
  rnorm(n, 0, 28), 1)

write.csv(escuelas, "datos/desempeno_escuelas.csv", row.names = FALSE)
cat("Dataset guardado: desempeno_escuelas.csv (", nrow(escuelas),
    "escuelas, puntaje medio", round(mean(escuelas$puntaje_prueba)), ")\n")
