# Script para crear el dataset de satisfacción democrática
# Datos simulados inspirados en encuestas como Latinobarómetro
# Se usa un dataset simulado para evitar dependencias de APIs externas en clase

library(tibble)
library(dplyr)

set.seed(2026)

paises <- c(
  "Argentina", "Bolivia", "Brasil", "Chile", "Colombia",
  "Costa Rica", "Ecuador", "El Salvador", "Guatemala", "Honduras",
  "México", "Nicaragua", "Panamá", "Paraguay", "Perú",
  "República Dominicana", "Uruguay", "Venezuela"
)

n_obs <- 500

datos <- tibble(
  id = 1:n_obs,
  pais = sample(paises, n_obs, replace = TRUE),
  edad = round(rnorm(n_obs, mean = 42, sd = 15)),
  educacion_anos = round(pmax(0, pmin(20, rnorm(n_obs, mean = 10, sd = 4)))),
  ingreso_hogar = round(pmax(100, rlnorm(n_obs, meanlog = 6.5, sdlog = 0.8))),
  confianza_gobierno = round(pmax(1, pmin(10, rnorm(n_obs, mean = 4.5, sd = 2.2)))),
  consumo_noticias = round(pmax(0, rnorm(n_obs, mean = 8, sd = 6)), 1),
  participacion_politica = round(pmax(0, pmin(100, rnorm(n_obs, mean = 35, sd = 25)))),
  zona = sample(c("urbano", "rural"), n_obs, replace = TRUE, prob = c(0.75, 0.25)),
  genero = sample(c("masculino", "femenino", "otro"), n_obs, replace = TRUE, prob = c(0.48, 0.50, 0.02))
)

# Ajustar edad para que sea realista (18-85)
datos$edad <- pmax(18, pmin(85, datos$edad))

# Crear variable de resultado: satisfacción con la democracia
# Con una relación real con los predictores
prob_satisfecho <- with(datos,
  plogis(-2.5 +
    0.01 * edad +
    0.05 * educacion_anos +
    0.0002 * ingreso_hogar +
    0.35 * confianza_gobierno +
    0.01 * consumo_noticias +
    0.015 * participacion_politica +
    ifelse(zona == "urbano", 0.2, 0) +
    rnorm(n_obs, 0, 0.5))
)

datos$satisfecho <- factor(
  ifelse(runif(n_obs) < prob_satisfecho, "si", "no"),
  levels = c("no", "si")
)

# Quitar columna id para el análisis
datos <- datos |> select(-id)

# Verificar balance
cat("Distribución del outcome:\n")
print(table(datos$satisfecho))
cat("\nProporción:\n")
print(prop.table(table(datos$satisfecho)))

# Guardar
write.csv(datos, "datos/satisfaccion_democracia.csv", row.names = FALSE)
cat("\nDataset guardado en datos/satisfaccion_democracia.csv\n")
cat("Dimensiones:", nrow(datos), "filas x", ncol(datos), "columnas\n")
