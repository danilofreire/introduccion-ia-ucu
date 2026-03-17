# Script para crear el dataset del laboratorio - Día 2
# Datos simulados inspirados en Latinobarómetro
# Variables de opinión pública en América Latina

library(tibble)

set.seed(2026)

paises <- c(
  "Argentina", "Bolivia", "Brasil", "Chile", "Colombia",
  "Costa Rica", "Ecuador", "El Salvador", "Guatemala", "Honduras",
  "México", "Nicaragua", "Panamá", "Paraguay", "Perú",
  "República Dominicana", "Uruguay", "Venezuela"
)

n_obs <- 500

datos <- tibble(
  pais = sample(paises, n_obs, replace = TRUE),
  edad = round(runif(n_obs, 18, 80)),
  educacion_anios = round(pmin(pmax(rnorm(n_obs, 10, 4), 0), 22)),
  ingreso_hogar = round(runif(n_obs, 1, 10)),  # decil 1-10
  zona = sample(c("urbana", "rural"), n_obs, replace = TRUE, prob = c(0.7, 0.3)),
  genero = sample(c("mujer", "hombre"), n_obs, replace = TRUE),
  confianza_gobierno = round(runif(n_obs, 1, 4)),  # 1=nada, 4=mucho
  confianza_justicia = round(runif(n_obs, 1, 4)),
  satisfaccion_democracia = round(runif(n_obs, 1, 4)),
  percepcion_economia = round(runif(n_obs, 1, 5)),  # 1=muy mala, 5=muy buena
  uso_internet = sample(c("diario", "semanal", "nunca"), n_obs, replace = TRUE, prob = c(0.5, 0.3, 0.2)),
  interes_politica = round(runif(n_obs, 1, 4))  # 1=nada, 4=mucho
)

# Variable continua para regresión: satisfacción con la vida (1-10)
datos$satisfaccion_vida <- with(datos,
  pmin(pmax(round(
    3.5 +
    0.08 * educacion_anios +
    0.15 * ingreso_hogar +
    0.3 * satisfaccion_democracia +
    0.2 * percepcion_economia -
    0.01 * edad +
    ifelse(zona == "urbana", 0.3, 0) +
    rnorm(n_obs, 0, 1.2),
    1), 1), 10)
)

# Variable binaria para clasificación: voto en la última elección
prob_voto <- with(datos,
  plogis(-2.8 +
    0.02 * edad +
    0.08 * educacion_anios +
    0.1 * interes_politica +
    0.05 * ingreso_hogar +
    ifelse(zona == "urbana", 0.3, 0) +
    0.15 * confianza_gobierno +
    rnorm(n_obs, 0, 0.3))
)

datos$voto <- factor(
  ifelse(runif(n_obs) < prob_voto, "si", "no"),
  levels = c("no", "si")
)

# Verificar balance
cat("Distribución de voto:\n")
print(table(datos$voto))
cat("\nProporción:\n")
print(prop.table(table(datos$voto)))

cat("\nResumen de satisfacción con la vida:\n")
print(summary(datos$satisfaccion_vida))

# Guardar
write.csv(datos, "datos/latinobarometro_sim.csv", row.names = FALSE)
cat("\nDataset guardado en datos/latinobarometro_sim.csv\n")
cat("Dimensiones:", nrow(datos), "filas x", ncol(datos), "columnas\n")
