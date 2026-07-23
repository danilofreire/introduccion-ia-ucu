# ============================================================
# IA para Científicos Sociales - UCU
# Laboratorio 5: Clustering y PCA
#
# Autor: Danilo Freire
# Fecha: mayo de 2026
#
# Este script contiene todo el código del Laboratorio 5.
# Cada sección coincide con una diapositiva (DEMO o EJERCICIO).
# Las soluciones de los ejercicios están incluidas tal como
# aparecen en los apéndices del .qmd.
# ============================================================


# --- Cargar paquetes -----------------------------------------

# Si algún paquete falta:
# install.packages(c("tidyverse", "cluster", "factoextra", "corrplot"))

library(tidyverse)
library(cluster)      # silhouette()
library(factoextra)   # fviz_cluster(), fviz_pca_*(), fviz_nbclust()
library(corrplot)     # matriz de correlación

set.seed(2026)


# --- Parte 1: Exploración de datos ---------------------------

# Cargar indicadores de 18 países latinoamericanos
paises <- read_csv("datos/indicadores_paises.csv", show_col_types = FALSE)
# Lean el archivo directo desde la web:
# paises <- read_csv("https://raw.githubusercontent.com/danilofreire/introduccion-ia-ucu/main/clases/dia-03/datos/indicadores_paises.csv", show_col_types = FALSE)

glimpse(paises)


# --- Estadísticas descriptivas -------------------------------

# Resumen general
paises |>
  select(-pais) |>
  summary()


# --- Matriz de correlación -----------------------------------

cor_matrix <- paises |>
  select(-pais) |>
  cor()

corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black", number.cex = 0.7,
         tl.col = "black", tl.srt = 45,
         col = colorRampPalette(c("#E74C3C", "white", "#2d4563"))(200))


# --- Ejercicio rápido: dos países (solución) -----------------

paises |>
  filter(pais %in% c("Uruguay", "Honduras")) |>
  select(pais, pib = pib_per_capita, internet = acceso_internet,
         democracia = indice_democracia, urban = urbanizacion,
         gini = indice_gini)

# Lecciones:
# - Uruguay tiene PIB per cápita más de 6 veces mayor que Honduras
# - Uruguay supera a Honduras en internet, democracia y urbanización
# - Pero Honduras tiene Gini más alto: es más desigual
# - El "desarrollo" no es una sola cosa


# --- Preparar datos para clustering --------------------------

# column_to_rownames("pais") mueve la columna pais a los nombres
# de fila, así scale() puede trabajar solo con las variables numéricas
# y los nombres de países se preservan automáticamente en los plots
datos_scaled <- paises |>
  column_to_rownames("pais") |>
  scale()

head(datos_scaled, 5)


# --- Parte 2: K-means y evaluación ---------------------------

# K-means con K=2 y 25 inicializaciones aleatorias
km2 <- kmeans(datos_scaled, centers = 2, nstart = 25)

# Asignación de clusters
km2$cluster

# Agregar al dataframe original
paises$cluster_km2 <- factor(km2$cluster)


# --- Visualizar los clusters ---------------------------------

fviz_cluster(km2, data = datos_scaled,
             palette = c("#2d4563", "#e63946"),
             geom = "point",
             ellipse.type = "convex",
             ggtheme = theme_minimal()) +
  geom_text(aes(label = rownames(datos_scaled)), vjust = -1, size = 3) +
  labs(title = "K-means con K=2")


# --- ¿Qué países en cada cluster? ----------------------------

paises |>
  group_by(cluster_km2) |>
  summarise(paises = paste(pais, collapse = ", "))


# --- Perfil de cada cluster ----------------------------------

paises |>
  group_by(cluster_km2) |>
  summarise(
    n = n(),
    pib_mean  = round(mean(pib_per_capita)),
    vida_mean = round(mean(esperanza_vida), 1),
    educ_mean = round(mean(anios_educacion), 1),
    inet_mean = round(mean(acceso_internet), 1),
    gini_mean = round(mean(indice_gini), 1),
    demo_mean = round(mean(indice_democracia), 1)
  )


# --- Método del codo -----------------------------------------

fviz_nbclust(datos_scaled, kmeans,
             method = "wss",
             k.max = 8,
             nstart = 25) +
  labs(title = "Método del codo",
       subtitle = "¿Dónde se 'aplana' la curva?") +
  theme_minimal()

# La mayor caída va de K=1 a K=2. El codo está en K=2.


# --- Método de la silueta ------------------------------------

fviz_nbclust(datos_scaled, kmeans,
             method = "silhouette",
             k.max = 8,
             nstart = 25) +
  labs(title = "Método de la silueta",
       subtitle = "K óptimo maximiza la silueta promedio") +
  theme_minimal()

# El pico está en K=2 (silueta ≈ 0.29). Codo y silueta coinciden.


# --- Silueta detallada para K=2 ------------------------------

sil <- silhouette(km2$cluster, dist(datos_scaled))

# Silueta promedio
mean(sil[, "sil_width"])


# --- Ejercicio rápido: K=3 (solución) ------------------------

km3  <- kmeans(datos_scaled, centers = 3, nstart = 25)
sil3 <- silhouette(km3$cluster, dist(datos_scaled))
mean(sil3[, "sil_width"])

# Países en cada cluster
data.frame(pais = rownames(datos_scaled), cluster = km3$cluster) |>
  arrange(cluster)

# Lecciones:
# - Silueta con K=3 ≈ 0.27 (con K=2 era ≈ 0.29, baja un poco)
# - El cluster "intermedio" suele tener países de ingreso medio
# - K=3 tiene sentido sustantivo aunque la silueta sea levemente peor


# --- Parte 3: PCA --------------------------------------------

# PCA sobre datos escalados (rownames se preservan)
pca <- prcomp(datos_scaled)

# Varianza explicada
summary(pca)


# --- Scree plot ----------------------------------------------

fviz_eig(pca, addlabels = TRUE,
         barfill = "#2d4563", barcolor = "#2d4563") +
  labs(title = "Scree plot: varianza explicada por componente") +
  theme_minimal()


# --- Loadings: ¿qué mide cada componente? --------------------

loadings <- pca$rotation[, 1:3] |>
  as.data.frame() |>
  rownames_to_column("variable") |>
  arrange(desc(abs(PC1)))

loadings

# Interpretación:
# - PC1 (52%): PIB, internet, democracia, urbanización, esperanza de vida,
#   educación cargan juntas (todas con signo negativo: un país con PC1 muy
#   negativo tiene más desarrollo). → eje de desarrollo general
# - PC2 (20%): Gini, esperanza de vida, gasto en salud altas; urbanización,
#   educación bajas. → contraste desigualdad vs. desarrollo urbano


# --- Ejercicio rápido: extremos de PC1 (solución) ------------

pca$x[, 1:2] |>
  as.data.frame() |>
  rownames_to_column("pais") |>
  arrange(PC1) |>
  mutate(PC1 = round(PC1, 2), PC2 = round(PC2, 2))

# Lecciones:
# - Valores más negativos de PC1: Uruguay, Chile, Argentina, Costa Rica
#   (los más "desarrollados" en este conjunto)
# - Valores más positivos: Honduras, Nicaragua, Guatemala (menos desarrollados)
# - Venezuela tiene el PC2 más bajo (-2.99): combina alta urbanización y
#   educación con el menor gasto en salud y la democracia más baja


# --- Biplot --------------------------------------------------

fviz_pca_biplot(pca,
                repel = TRUE,
                col.var = "#e63946",
                col.ind = "#2d4563",
                label = "all") +
  labs(title = "Biplot: países y variables") +
  theme_minimal()


# ============================================================
# MATERIAL OPCIONAL (apéndices del laboratorio)
# Estas secciones no se cubren en clase. Se incluyen para
# quienes quieran profundizar en casa.
# ============================================================


# --- Apéndice 1: Clustering jerárquico -----------------------

# Matriz de distancias euclidianas
d <- dist(datos_scaled, method = "euclidean")

# Clustering jerárquico con método Ward
hc <- hclust(d, method = "ward.D2")

# Dendrograma con 2 cortes
plot(hc, hang = -1, cex = 0.9,
     main = "Dendrograma: Países latinoamericanos",
     xlab = "País", ylab = "Altura")
rect.hclust(hc, k = 2, border = c("#2d4563", "#e63946"))

# Cortar para obtener K=2 clusters
grupos_hc <- cutree(hc, k = 2)

# Comparar con K-means
table(K_means = km2$cluster, Jerarquico = grupos_hc)


# --- Apéndice 2: PCA coloreado por cluster K-means -----------

pca_coords <- pca$x[, 1:2] |>
  as.data.frame() |>
  rownames_to_column("pais") |>
  mutate(cluster = paises$cluster_km2)

ggplot(pca_coords, aes(x = PC1, y = PC2, color = cluster, label = pais)) +
  geom_point(size = 3) +
  geom_text(vjust = -0.8, size = 3, show.legend = FALSE) +
  scale_color_manual(values = c("#2d4563", "#e63946")) +
  labs(title = "PCA coloreado por cluster K-means",
       x = paste0("PC1 (", round(summary(pca)$importance[2, 1] * 100, 1), "%)"),
       y = paste0("PC2 (", round(summary(pca)$importance[2, 2] * 100, 1), "%)")) +
  theme_minimal()


# --- Apéndice 2 (cont.): Contribución de variables a PC1 -----

fviz_contrib(pca, choice = "var", axes = 1, fill = "#2d4563") +
  labs(title = "Contribución de variables a PC1") +
  theme_minimal()


# --- Apéndice 3: Ejercicio adicional — PCA con 3 componentes -

pca_3d <- pca$x[, 1:3] |>
  as.data.frame() |>
  rownames_to_column("pais") |>
  mutate(cluster = paises$cluster_km2)

ggplot(pca_3d, aes(x = PC1, y = PC3, color = cluster, label = pais)) +
  geom_point(size = 3) +
  geom_text(vjust = -0.8, size = 3, show.legend = FALSE) +
  scale_color_manual(values = c("#2d4563", "#e63946")) +
  labs(title = "PC1 vs PC3") +
  theme_minimal()


# --- Apéndice 3: Ejercicio adicional — K=4 -------------------

km4 <- kmeans(datos_scaled, centers = 4, nstart = 25)

# Silueta promedio
sil4 <- silhouette(km4$cluster, dist(datos_scaled))
mean(sil4[, "sil_width"])

# Visualizar
fviz_cluster(km4, data = datos_scaled,
             palette = c("#2d4563", "#e63946", "#457b9d", "#f4a261"),
             ggtheme = theme_minimal()) +
  labs(title = "K-means con K=4")
