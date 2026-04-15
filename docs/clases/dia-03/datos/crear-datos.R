# Script para crear los datasets del Día 3
# Dataset 1: indicadores por país para clustering/PCA
# Dataset 2: corpus de textos para análisis de texto

library(tibble)

set.seed(2026)

# --- Dataset 1: Indicadores socioeconómicos por país ---

paises <- c(
  "Argentina", "Bolivia", "Brasil", "Chile", "Colombia",
  "Costa Rica", "Ecuador", "El Salvador", "Guatemala", "Honduras",
  "México", "Nicaragua", "Panamá", "Paraguay", "Perú",
  "República Dominicana", "Uruguay", "Venezuela"
)

n <- length(paises)

indicadores <- tibble(
  pais = paises,
  pib_per_capita = round(c(10600, 3600, 8900, 15400, 6100,
                           12500, 6200, 4400, 5000, 2800,
                           10400, 2100, 15800, 5800, 7000,
                           8500, 17000, 3700) + rnorm(n, 0, 300)),
  esperanza_vida = round(c(76.5, 71.5, 75.9, 80.2, 77.3,
                           80.3, 77.0, 73.1, 74.1, 75.1,
                           75.1, 74.5, 78.5, 74.0, 76.5,
                           74.0, 78.0, 72.1) + rnorm(n, 0, 0.3), 1),
  anios_educacion = round(c(10.6, 9.0, 8.0, 10.3, 8.3,
                            8.7, 8.3, 7.0, 5.6, 6.6,
                            8.6, 6.5, 10.2, 8.1, 9.2,
                            7.6, 8.6, 10.3) + rnorm(n, 0, 0.2), 1),
  acceso_internet = round(c(87.1, 53.4, 81.3, 88.3, 73.0,
                            81.2, 64.3, 50.5, 44.5, 39.0,
                            75.6, 30.6, 65.0, 68.8, 65.3,
                            74.8, 87.7, 72.0) + rnorm(n, 0, 1), 1),
  gasto_salud_pib = round(c(10.0, 6.9, 9.6, 9.3, 7.7,
                            7.9, 8.3, 7.2, 5.8, 7.4,
                            5.4, 8.6, 7.3, 7.0, 5.2,
                            5.5, 9.2, 3.7) + rnorm(n, 0, 0.2), 1),
  indice_gini = round(c(42.3, 43.6, 48.9, 44.9, 51.3,
                        48.2, 45.4, 38.8, 48.3, 52.1,
                        45.4, 46.2, 49.8, 46.2, 41.5,
                        39.6, 39.7, 44.7) + rnorm(n, 0, 0.5), 1),
  urbanizacion = round(c(92.1, 70.1, 87.1, 87.7, 81.4,
                         80.8, 64.2, 73.4, 51.8, 58.4,
                         80.7, 59.0, 68.1, 62.2, 78.3,
                         82.5, 95.5, 88.3) + rnorm(n, 0, 0.3), 1),
  indice_democracia = round(c(6.8, 4.4, 6.9, 8.0, 7.0,
                              8.0, 5.6, 5.8, 4.9, 5.4,
                              5.6, 3.3, 6.6, 6.2, 6.7,
                              6.5, 8.6, 2.2) + rnorm(n, 0, 0.1), 1)
)

write.csv(indicadores, "datos/indicadores_paises.csv", row.names = FALSE)
cat("Dataset 1 guardado: indicadores_paises.csv (", nrow(indicadores), "países)\n")

# --- Dataset 2: Corpus de textos políticos simulados ---

temas <- c("economia", "educacion", "seguridad", "salud", "medioambiente")
sentimientos <- c("positivo", "negativo", "neutro")

textos <- tibble(
  id = 1:60,
  pais = sample(paises, 60, replace = TRUE),
  tema = rep(temas, each = 12),
  anio = sample(2018:2024, 60, replace = TRUE),
  texto = c(
    # Economía (12 textos)
    "El crecimiento económico del país ha sido sólido este año gracias a las exportaciones agrícolas y la inversión extranjera directa.",
    "La inflación sigue siendo un problema grave que afecta a las familias más vulnerables del país.",
    "El gobierno anunció un nuevo plan de estímulo fiscal para reactivar la economía después de la pandemia.",
    "Las tasas de desempleo han alcanzado niveles históricos y el mercado laboral no muestra signos de recuperación.",
    "La política monetaria del banco central ha logrado estabilizar la moneda frente a las presiones externas.",
    "Los acuerdos comerciales con Asia representan una oportunidad para diversificar nuestras exportaciones.",
    "La deuda externa del país ha crecido significativamente poniendo en riesgo la estabilidad fiscal.",
    "El sector tecnológico emerge como un nuevo motor de crecimiento económico en la región.",
    "Las remesas de los emigrantes representan una fuente vital de ingresos para miles de familias.",
    "La informalidad laboral sigue siendo uno de los mayores desafíos económicos del continente.",
    "Los programas de transferencia condicionada han demostrado ser efectivos para reducir la pobreza extrema.",
    "La inversión en infraestructura es insuficiente para sostener el desarrollo económico a largo plazo.",
    # Educación (12 textos)
    "La inversión en educación pública ha permitido aumentar la cobertura escolar en las zonas rurales.",
    "La brecha digital entre estudiantes urbanos y rurales se ha profundizado durante la pandemia.",
    "El nuevo programa de becas universitarias beneficiará a miles de estudiantes de bajos recursos.",
    "La calidad educativa en las escuelas públicas sigue siendo inferior a la de las instituciones privadas.",
    "Los docentes exigen mejores salarios y condiciones laborales para mejorar la enseñanza.",
    "La educación técnica y profesional necesita mayor apoyo para formar trabajadores calificados.",
    "Las universidades públicas han implementado programas innovadores de educación a distancia.",
    "El analfabetismo funcional afecta a millones de adultos en la región impidiendo su participación plena.",
    "Los programas de alimentación escolar han mejorado la asistencia y el rendimiento académico.",
    "La formación docente requiere una reforma urgente para adaptarse a las necesidades del siglo XXI.",
    "La educación bilingüe para comunidades indígenas ha avanzado pero aún es insuficiente.",
    "Las pruebas estandarizadas muestran que los estudiantes latinoamericanos están por debajo del promedio mundial.",
    # Seguridad (12 textos)
    "Las políticas de mano dura contra el crimen organizado no han logrado reducir la violencia en la región.",
    "El gobierno ha implementado programas de prevención social para jóvenes en riesgo de violencia.",
    "La tasa de homicidios ha disminuido por tercer año consecutivo gracias a las reformas policiales.",
    "El narcotráfico sigue siendo la principal amenaza para la seguridad en América Central.",
    "La cooperación regional en materia de seguridad ha permitido desarticular redes criminales transnacionales.",
    "La violencia de género ha aumentado y los sistemas de justicia no responden adecuadamente.",
    "Las fuerzas de seguridad necesitan mayor capacitación en derechos humanos y uso proporcional de la fuerza.",
    "Los programas de justicia restaurativa han mostrado resultados prometedores en la reducción de la reincidencia.",
    "La corrupción policial sigue siendo un obstáculo para mejorar la seguridad ciudadana.",
    "Las cárceles están sobrepobladas y no cumplen su función de rehabilitación social.",
    "La extorsión a pequeños comerciantes es un problema creciente en las ciudades principales.",
    "Los sistemas de emergencia y respuesta rápida han mejorado la atención a las víctimas de delitos.",
    # Salud (12 textos)
    "El sistema de salud pública ha sido fortalecido con nuevos hospitales y centros de atención primaria.",
    "La pandemia expuso las graves deficiencias del sistema sanitario en toda la región.",
    "Los programas de vacunación han alcanzado coberturas históricas gracias a la coordinación internacional.",
    "La desigualdad en el acceso a servicios de salud entre zonas urbanas y rurales es alarmante.",
    "El gasto de bolsillo en salud empuja a millones de familias a la pobreza cada año.",
    "La telemedicina ha emergido como una herramienta valiosa para llegar a comunidades remotas.",
    "Las enfermedades crónicas no transmisibles representan una carga creciente para los sistemas de salud.",
    "La salud mental sigue siendo un tema desatendido en las políticas públicas de la región.",
    "Los trabajadores de salud enfrentan condiciones precarias y salarios insuficientes.",
    "La investigación biomédica en la región necesita mayor financiamiento y cooperación internacional.",
    "Los programas de salud materno-infantil han reducido significativamente la mortalidad neonatal.",
    "El acceso a medicamentos esenciales sigue siendo un desafío para las poblaciones más vulnerables.",
    # Medio ambiente (12 textos)
    "La deforestación en la Amazonia ha alcanzado niveles récord amenazando la biodiversidad global.",
    "Las energías renovables representan una oportunidad de desarrollo sostenible para la región.",
    "Los incendios forestales han devastado miles de hectáreas de bosque nativo en los últimos años.",
    "La contaminación del agua afecta a comunidades rurales que dependen de fuentes naturales.",
    "Los países de la región han asumido compromisos ambiciosos en la lucha contra el cambio climático.",
    "La minería ilegal destruye ecosistemas frágiles y amenaza la salud de las comunidades locales.",
    "Los humedales y manglares están desapareciendo a un ritmo alarmante en toda la costa.",
    "La transición energética requiere inversiones significativas en infraestructura y capacitación.",
    "La agricultura sostenible puede reducir las emisiones de carbono sin sacrificar la productividad.",
    "Las comunidades indígenas son guardianes esenciales de la biodiversidad en la región.",
    "La gestión de residuos sólidos es deficiente en la mayoría de las ciudades latinoamericanas.",
    "Los eventos climáticos extremos son cada vez más frecuentes y afectan desproporcionadamente a los más pobres."
  )
)

write.csv(textos, "datos/textos_politicos.csv", row.names = FALSE)
cat("Dataset 2 guardado: textos_politicos.csv (", nrow(textos), "textos)\n")
