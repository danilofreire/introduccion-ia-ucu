# Script para crear el dataset del laboratorio
# Datos inspirados en indicadores del Banco Mundial para países de América Latina
# Se usa un dataset simulado para evitar dependencias de APIs externas en clase

library(tibble)

set.seed(2026)

paises <- c(
  "Argentina", "Bolivia", "Brasil", "Chile", "Colombia",
  "Costa Rica", "Ecuador", "El Salvador", "Guatemala", "Honduras",
  "México", "Nicaragua", "Panamá", "Paraguay", "Perú",
  "República Dominicana", "Uruguay", "Venezuela"
)

n_obs <- 180 # 10 observaciones por país (simulando años)

datos <- tibble(
  pais = sample(paises, n_obs, replace = TRUE),
  anio = sample(2005:2023, n_obs, replace = TRUE),
  gasto_educacion = round(runif(n_obs, 2.5, 7.5), 1),
  acceso_internet = round(runif(n_obs, 15, 90), 1),
  urbanizacion = round(runif(n_obs, 45, 95), 1),
  gasto_salud = round(runif(n_obs, 3.0, 9.5), 1),
  inflacion = round(rexp(n_obs, rate = 0.15) + 1, 1),
  desempleo = round(runif(n_obs, 2.5, 15), 1),
  inversion_extranjera = round(runif(n_obs, 0.5, 8.0), 1),
  indice_gobierno_digital = round(runif(n_obs, 0.2, 0.9), 2)
)

# Crear variable de resultado: crecimiento alto (>= 3%) vs bajo (< 3%)
# Con una relación real con los predictores
prob_alto <- with(datos,
  plogis(-3.8 +
    0.15 * gasto_educacion +
    0.02 * acceso_internet +
    0.01 * urbanizacion +
    0.1 * gasto_salud -
    0.03 * inflacion -
    0.05 * desempleo +
    0.08 * inversion_extranjera +
    1.5 * indice_gobierno_digital +
    rnorm(n_obs, 0, 0.3))
)

datos$crecimiento_alto <- factor(
  ifelse(runif(n_obs) < prob_alto, "si", "no"),
  levels = c("no", "si")
)

# Verificar balance
cat("Distribución del outcome:\n")
print(table(datos$crecimiento_alto))
cat("\nProporción:\n")
print(prop.table(table(datos$crecimiento_alto)))

# Guardar
write.csv(datos, "datos/indicadores_latam.csv", row.names = FALSE)
cat("\nDataset guardado en datos/indicadores_latam.csv\n")
cat("Dimensiones:", nrow(datos), "filas x", ncol(datos), "columnas\n")
