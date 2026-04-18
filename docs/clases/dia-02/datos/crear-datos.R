# Script para crear el dataset simulado de los Laboratorios 3 y 4
# Encuesta tipo Latinobarometro: 500 encuestados de 18 paises latinoamericanos
#
# Diseno:
#   - Las variables de actitud politica estan correlacionadas entre si
#     (interes politico -> confianza gobierno -> confianza justicia).
#   - Ingreso correlaciona con educacion (~0.30).
#   - Outcome Lab 3 (voto): 11 predictores con efecto directo. Efectos mas
#     fuertes: interes politico, edad, confianza gobierno. Efecto negativo:
#     percepcion economica (voto de protesta). Ruido sd = 1.0 -> ROC-AUC ~0.72-0.78.
#   - Outcome Lab 4 (satisfaccion_vida): funcion lineal de educacion, ingreso,
#     confianza en el gobierno, satisfaccion con la democracia, percepcion
#     economica e interes politico. Ruido sd = 1.2 -> R^2 esperado ~0.35-0.45.

set.seed(2026)
n <- 500

# --- Paises latinoamericanos ---

paises <- c(
  "Argentina", "Bolivia", "Brasil", "Chile", "Colombia",
  "Costa Rica", "Ecuador", "El Salvador", "Guatemala",
  "Honduras", "México", "Nicaragua", "Panamá", "Paraguay",
  "Perú", "República Dominicana", "Uruguay", "Venezuela"
)

# --- Variables demograficas ---

pais <- sample(paises, n, replace = TRUE)
edad <- round(runif(n, 18, 85))
educacion_anios <- round(pmin(pmax(rnorm(n, 11, 4), 0), 20))
# Ingreso correlacionado con educacion (mas educacion -> mas ingreso)
ingreso_hogar <- round(pmin(pmax(0.3 * educacion_anios + rnorm(n, 2, 2), 1), 10))
zona <- sample(c("urbana", "rural"), n, replace = TRUE, prob = c(0.70, 0.30))
genero <- sample(c("hombre", "mujer"), n, replace = TRUE)

# --- Actitudes politicas (correlacionadas entre si) ---

interes_politica <- round(pmin(pmax(rnorm(n, 2.5, 1), 1), 5))
# Confianza en gobierno correlacionada con interes politico
confianza_gobierno <- round(pmin(pmax(
  0.3 * interes_politica + rnorm(n, 1.5, 0.8), 1), 5))
# Confianza en justicia correlacionada con confianza en gobierno
confianza_justicia <- round(pmin(pmax(
  0.2 * confianza_gobierno + rnorm(n, 2, 0.9), 1), 5))
# Satisfaccion con democracia correlacionada con confianza en gobierno
satisfaccion_democracia <- round(pmin(pmax(
  0.2 * confianza_gobierno + rnorm(n, 2, 0.9), 1), 5))
percepcion_economia <- round(pmin(pmax(rnorm(n, 3, 1.2), 1), 5))

# --- Otras variables ---

uso_internet <- sample(c("nunca", "semanal", "diario"), n, replace = TRUE,
                        prob = c(0.15, 0.30, 0.55))
satisfaccion_vida <- round(pmin(pmax(
  2.0 +
  0.08 * educacion_anios +          # mas educacion -> mas satisfaccion
  0.15 * ingreso_hogar +            # mas ingreso -> mas satisfaccion
  0.35 * confianza_gobierno +       # confianza en gobierno -> satisfaccion
  0.30 * satisfaccion_democracia +  # satisfaccion democratica -> vital
  0.20 * percepcion_economia +      # buena economia -> mas satisfaccion
  0.15 * interes_politica +         # interes politico -> efecto leve
  0.01 * edad +                     # edad -> efecto muy leve
  rnorm(n, 0, 1.2),                 # ruido residual
  1), 10), 1)

# --- Outcome: voto en la ultima eleccion ---

# Codificar variables categoricas para el modelo generador
internet_num <- ifelse(uso_internet == "diario", 1,
                       ifelse(uso_internet == "semanal", 0, -1))
zona_num <- ifelse(zona == "urbana", 0.5, -0.5)

# Predictor lineal latente: los coeficientes definen la
# fuerza real de cada variable sobre la probabilidad de votar.
# plogis() convierte el valor lineal en probabilidad (0 a 1).
logit_voto <- -4.5 +
  0.025 * edad +                     # Mas edad -> mas voto
  0.08  * educacion_anios +          # Mas educacion -> mas voto
  0.05  * ingreso_hogar +            # Mas ingreso -> mas voto
  0.15  * zona_num +                 # Zona urbana -> algo mas
  0.25  * confianza_gobierno +       # Confianza en gobierno -> voto
  0.10  * confianza_justicia +       # Confianza en justicia -> voto
  0.15  * satisfaccion_democracia +  # Satisfaccion democratica -> voto
 -0.08  * percepcion_economia +      # Mala economia -> voto de protesta
  0.20  * internet_num +             # Uso de internet -> mas participacion
  0.45  * interes_politica +         # Interes politico -> efecto fuerte
  0.03  * satisfaccion_vida +        # Satisfaccion vital -> efecto debil
  rnorm(n, 0, 1.0)                   # Ruido aleatorio

prob_voto <- plogis(logit_voto)
voto <- ifelse(runif(n) < prob_voto, "si", "no")

# --- Crear data.frame y guardar ---

datos <- data.frame(
  pais, edad, educacion_anios, ingreso_hogar, zona, genero,
  confianza_gobierno, confianza_justicia, satisfaccion_democracia,
  percepcion_economia, uso_internet, interes_politica,
  satisfaccion_vida, voto
)

# --- Verificaciones ---

cat("Dimensiones:", nrow(datos), "filas x", ncol(datos), "columnas\n")

cat("\nDistribucion del outcome:\n")
print(table(datos$voto))
cat("\nProporcion:\n")
print(round(prop.table(table(datos$voto)), 3))

cat("\nCorrelaciones entre predictores numericos:\n")
num_cols <- datos[, c("edad", "educacion_anios", "ingreso_hogar",
                      "confianza_gobierno", "confianza_justicia",
                      "satisfaccion_democracia", "percepcion_economia",
                      "interes_politica", "satisfaccion_vida")]
print(round(cor(num_cols), 2))

cat("\nVerificacion - modelo logistico:\n")
modelo <- glm(factor(voto) ~ edad + educacion_anios + ingreso_hogar +
              zona + genero + confianza_gobierno + confianza_justicia +
              satisfaccion_democracia + percepcion_economia +
              uso_internet + interes_politica + satisfaccion_vida,
              data = datos, family = binomial())
print(round(summary(modelo)$coefficients, 4))

# Guardar
write.csv(datos, "latinobarometro_sim.csv", row.names = FALSE)
cat("\nDataset guardado en datos/latinobarometro_sim.csv\n")
