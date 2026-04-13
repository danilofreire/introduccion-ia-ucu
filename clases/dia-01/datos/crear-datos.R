# Script para crear el dataset del laboratorio
# Corte transversal de ~195 países del mundo (datos tipo Banco Mundial, año 2024)
#
# Diseño:
#   - Un factor latente de "desarrollo" por país genera correlaciones moderadas
#     (~0.15-0.30) entre predictores. Cada predictor tiene bastante ruido propio.
#   - 6 de 8 predictores tienen efecto directo sobre el outcome.
#   - Urbanización y gasto en salud NO tienen efecto directo (buen ejercicio
#     para discutir por qué un predictor puede correlacionar con Y sin ser causal).

library(tibble)

set.seed(2026)

# --- Países del mundo organizados por continente ---

africa <- c(
  "Angola", "Argelia", "Benín", "Botsuana", "Burkina Faso",
  "Burundi", "Cabo Verde", "Camerún", "Chad", "Comoras",
  "Costa de Marfil", "Egipto", "Eritrea", "Etiopía", "Gabón",
  "Gambia", "Ghana", "Guinea", "Guinea-Bisáu", "Guinea Ecuatorial",
  "Kenia", "Lesoto", "Liberia", "Libia", "Madagascar",
  "Malaui", "Malí", "Marruecos", "Mauricio", "Mauritania",
  "Mozambique", "Namibia", "Níger", "Nigeria", "Rep. Centroafricana",
  "Rep. del Congo", "Rep. Dem. del Congo", "Ruanda", "Senegal",
  "Sierra Leona", "Somalia", "Sudáfrica", "Sudán", "Sudán del Sur",
  "Suazilandia", "Tanzania", "Togo", "Túnez", "Uganda",
  "Yibuti", "Zambia", "Zimbabue"
)

americas <- c(
  "Argentina", "Bahamas", "Barbados", "Belice", "Bolivia",
  "Brasil", "Canadá", "Chile", "Colombia", "Costa Rica",
  "Cuba", "Ecuador", "El Salvador", "Estados Unidos", "Guatemala",
  "Guyana", "Haití", "Honduras", "Jamaica", "México",
  "Nicaragua", "Panamá", "Paraguay", "Perú", "Rep. Dominicana",
  "Surinam", "Trinidad y Tobago", "Uruguay", "Venezuela"
)

asia <- c(
  "Afganistán", "Arabia Saudita", "Armenia", "Azerbaiyán",
  "Bangladés", "Birmania", "Brunéi", "Bután", "Camboya", "China",
  "Corea del Norte", "Corea del Sur", "Emiratos Árabes", "Filipinas",
  "Georgia", "India", "Indonesia", "Irak", "Irán", "Israel",
  "Japón", "Jordania", "Kazajistán", "Kirguistán", "Kuwait",
  "Laos", "Líbano", "Malasia", "Maldivas", "Mongolia",
  "Nepal", "Omán", "Pakistán", "Qatar", "Singapur",
  "Siria", "Sri Lanka", "Tailandia", "Tayikistán", "Timor Oriental",
  "Turkmenistán", "Turquía", "Uzbekistán", "Vietnam", "Yemen"
)

europa <- c(
  "Albania", "Alemania", "Austria", "Bélgica", "Bielorrusia",
  "Bosnia y Herzegovina", "Bulgaria", "Chipre", "Croacia",
  "Dinamarca", "Eslovaquia", "Eslovenia", "España", "Estonia",
  "Finlandia", "Francia", "Grecia", "Hungría", "Irlanda",
  "Islandia", "Italia", "Letonia", "Lituania", "Luxemburgo",
  "Macedonia del Norte", "Malta", "Moldavia", "Montenegro",
  "Noruega", "Países Bajos", "Polonia", "Portugal", "Reino Unido",
  "Rep. Checa", "Rumania", "Rusia", "Serbia", "Suecia",
  "Suiza", "Ucrania"
)

oceania <- c(
  "Australia", "Fiyi", "Islas Marshall", "Islas Salomón",
  "Kiribati", "Micronesia", "Nueva Zelanda", "Palaos",
  "Papúa Nueva Guinea", "Samoa", "Tonga", "Tuvalu", "Vanuatu"
)

# Construir tabla de países con continente
pais_info <- data.frame(
  pais = c(africa, americas, asia, europa, oceania),
  continente = c(
    rep("África", length(africa)),
    rep("América", length(americas)),
    rep("Asia", length(asia)),
    rep("Europa", length(europa)),
    rep("Oceanía", length(oceania))
  ),
  stringsAsFactors = FALSE
)

n <- nrow(pais_info)
cat("Total de países:", n, "\n")

# --- Factor latente de desarrollo ---
# Distribución por continente: refleja diferencias reales en IDH promedio.
# El ruido dentro de cada continente (sd grande) genera la variación
# entre países de una misma región (ej. Japón vs. Yemen en Asia).
desarrollo <- with(pais_info, ifelse(
  continente == "Europa",  rnorm(n,  0.8, 0.5),
  ifelse(continente == "América", rnorm(n,  0.1, 0.6),
  ifelse(continente == "Asia",    rnorm(n,  0.0, 0.7),
  ifelse(continente == "Oceanía", rnorm(n,  0.0, 0.8),
                                  rnorm(n, -0.6, 0.5))))))  # África

# --- Predictores ---
# Loadings bajos en el factor latente + ruido grande = correlaciones ~0.15-0.30

datos <- tibble(
  pais       = pais_info$pais,
  continente = pais_info$continente,

  gasto_educacion = round(pmax(2.0, pmin(8.5,
    5.0 + 0.5 * desarrollo + rnorm(n, 0, 1.0)
  )), 1),

  acceso_internet = round(pmax(5, pmin(99,
    55 + 10 * desarrollo + rnorm(n, 0, 15)
  )), 1),

  urbanizacion = round(pmax(15, pmin(100,
    60 + 8 * desarrollo + rnorm(n, 0, 15)
  )), 1),

  gasto_salud = round(pmax(2.5, pmin(12.0,
    6.5 + 0.5 * desarrollo + rnorm(n, 0, 1.2)
  )), 1),

  inflacion = round(pmax(0.5, pmin(40,
    8 - 2.0 * desarrollo + rexp(n, rate = 0.2)
  )), 1),

  desempleo = round(pmax(1.5, pmin(28,
    9.0 - 1.5 * desarrollo + rnorm(n, 0, 3.0)
  )), 1),

  inversion_extranjera = round(pmax(0.2, pmin(12,
    4.0 + 0.8 * desarrollo + rnorm(n, 0, 1.5)
  )), 1),

  indice_gobierno_digital = round(pmax(0.05, pmin(0.98,
    0.50 + 0.12 * desarrollo + rnorm(n, 0, 0.15)
  )), 2)
)

# --- Outcome ---
# Crecimiento alto del PIB (>= 3%)
# 6 de 8 predictores tienen efecto directo; urbanización y gasto_salud NO.
prob_alto <- with(datos,
  plogis(-7.5 +
    0.60 * gasto_educacion +           # (+) positivo moderado
    0.06 * acceso_internet +           # (+) positivo
    0.00 * urbanizacion +              # (0) SIN efecto directo
    0.00 * gasto_salud -               # (0) SIN efecto directo
    0.18 * inflacion -                 # (-) negativo
    0.22 * desempleo +                 # (-) negativo
    0.45 * inversion_extranjera +      # (+) positivo moderado
    6.00 * indice_gobierno_digital +   # (+) positivo fuerte
    rnorm(n, 0, 0.3))                  # ruido residual pequeño
)

datos$crecimiento_alto <- factor(
  ifelse(runif(n) < prob_alto, "si", "no"),
  levels = c("no", "si")
)

# --- Verificaciones ---

cat("Dimensiones:", nrow(datos), "filas x", ncol(datos), "columnas\n")
cat("Países por continente:\n")
print(table(datos$continente))

cat("\nDistribución del outcome:\n")
print(table(datos$crecimiento_alto))
cat("\nProporción:\n")
print(round(prop.table(table(datos$crecimiento_alto)), 3))

cat("\nCorrelaciones entre predictores:\n")
num_cols <- datos[, c("gasto_educacion", "acceso_internet", "urbanizacion",
                      "gasto_salud", "inflacion", "desempleo",
                      "inversion_extranjera", "indice_gobierno_digital")]
print(round(cor(num_cols), 2))

cat("\nVerificación — modelo logístico:\n")
modelo <- glm(crecimiento_alto ~ gasto_educacion + acceso_internet +
              urbanizacion + gasto_salud + inflacion + desempleo +
              inversion_extranjera + indice_gobierno_digital,
              data = datos, family = binomial())
print(round(summary(modelo)$coefficients, 4))

# Guardar
write.csv(datos, "datos/indicadores_mundiales.csv", row.names = FALSE)
cat("\nDataset guardado en datos/indicadores_mundiales.csv\n")
