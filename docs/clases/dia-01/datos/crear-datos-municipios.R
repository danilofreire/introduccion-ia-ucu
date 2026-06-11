# Script para crear el dataset de la Tarea 1
# Corte transversal de 300 municipios latinoamericanos (datos simulados)
#
# Diseño:
#   - Un factor latente de "desarrollo" por municipio genera correlaciones
#     moderadas entre predictores. Cada predictor tiene bastante ruido propio.
#   - 5 de 8 predictores tienen efecto directo sobre el outcome.
#   - acceso_electricidad y poblacion NO tienen efecto directo (correlacionan
#     con el outcome vía el factor latente: buen ejercicio para discutir por
#     qué un predictor puede correlacionar con Y sin ser causal).

library(tibble)

set.seed(2026)

paises <- c("Argentina", "Brasil", "Colombia", "México", "Perú", "Uruguay")

n <- 300

# Factor latente de desarrollo de cada municipio
d <- rnorm(n)

municipios <- tibble(
  municipio = sprintf("municipio_%03d", 1:n),
  pais = sample(paises, n, replace = TRUE),
  ingreso_promedio = round(900 + 350 * d + rnorm(n, 0, 250)),
  escolaridad_media = round(8.5 + 1.6 * d + rnorm(n, 0, 1.2), 1),
  acceso_agua = round(pmin(100, pmax(20, 78 + 11 * d + rnorm(n, 0, 9))), 1),
  acceso_electricidad = round(pmin(100, pmax(35, 88 + 7 * d + rnorm(n, 0, 6))), 1),
  empleo_formal = round(pmin(95, pmax(5, 45 + 13 * d + rnorm(n, 0, 10))), 1),
  distancia_capital = round(pmax(5, 240 - 40 * d + rnorm(n, 0, 130))),
  poblacion = round(pmax(2, exp(3.4 + 0.45 * d + rnorm(n, 0, 0.9)))),
  presupuesto_per_capita = round(pmax(50, 320 + 60 * d + rnorm(n, 0, 110)))
)

# Pisos mínimos realistas
municipios$ingreso_promedio <- pmax(250, municipios$ingreso_promedio)

# Outcome: desarrollo alto (si/no)
# Efectos directos: ingreso, escolaridad, agua, empleo, presupuesto y (débil,
# negativa) distancia. Electricidad y población quedan FUERA a propósito.
lp <- with(municipios,
  -0.35 +
  0.9 * scale(ingreso_promedio)[, 1] +
  0.7 * scale(escolaridad_media)[, 1] +
  0.6 * scale(acceso_agua)[, 1] +
  0.5 * scale(empleo_formal)[, 1] +
  0.35 * scale(presupuesto_per_capita)[, 1] -
  0.3 * scale(distancia_capital)[, 1] +
  rnorm(n, 0, 1)
)

municipios$desarrollo_alto <- ifelse(runif(n) < plogis(lp), "si", "no")

write.csv(municipios, "datos/desarrollo_municipios.csv", row.names = FALSE)
cat("Dataset guardado: desarrollo_municipios.csv (", nrow(municipios),
    "municipios,", round(mean(municipios$desarrollo_alto == "si") * 100),
    "% desarrollo alto)\n")
