# Introducción a la IA para Científicos Sociales

Curso intensivo de introducción a la Inteligencia Artificial y el Machine Learning para estudiantes de ciencias sociales, dictado en R. Este repositorio contiene las diapositivas, los laboratorios, las tareas y las lecturas del curso.

**Sitio web:** <https://danilofreire.github.io/introduccion-ia-ucu>

## Sobre el curso

|  |  |
|--|--|
| **Docente** | Danilo Freire |
| **Institución** | Universidad Católica del Uruguay |
| **Programa** | Escuela en Métodos, Centro Hodos |
| **Modalidad** | Presencial, 5 días (20 horas) |
| **Idioma** | Español |
| **Herramientas** | R, RStudio, tidyverse, tidymodels, Ollama |

El curso introduce los fundamentos de la IA y el Machine Learning con un enfoque aplicado: priorizamos la intuición y el uso de herramientas computacionales sobre la teoría matemática. Está pensado para quienes ya tienen una base en R y quieren expandir sus capacidades hacia métodos computacionales modernos. No se requiere conocimiento de Python ni de matemáticas avanzadas.

## Contenido por día

| Día | Tema | Técnicas y paquetes |
|-----|------|---------------------|
| 1 | Fundamentos de IA y ML | Flujo de trabajo, train/test, validación cruzada, métricas; `tidymodels` |
| 2 | Aprendizaje supervisado | Regresión logística, árboles, Random Forest, regularización; `ranger`, `glmnet` |
| 3 | Texto y aprendizaje no supervisado | K-means, PCA, tokenización, TF-IDF, sentimiento, LDA; `tidytext`, `topicmodels` |
| 4 | Modelos de lenguaje (LLMs) | Transformers, embeddings, prompt engineering, RAG, anotación; `ellmer`, `quallmer` |
| 5 | Modelos locales, ética y cierre | Ollama, sesgo algorítmico, auditoría de equidad, regulación; `fairmodels`, `DALEX` |

Cada día combina dos sesiones teóricas con laboratorios prácticos en R, más una tarea para reforzar lo visto.

## Estructura del repositorio

```
.
├── index.qmd            # Página de inicio
├── programa.qmd         # Programa completo y requisitos de software
├── laboratorios.qmd     # Laboratorios y tareas
├── lecturas.qmd         # Lecturas obligatorias y complementarias
├── referencia-*.qmd     # Páginas de referencia rápida de R
├── clases/              # Diapositivas, laboratorios y tareas, por día
│   └── dia-01 … dia-05
├── custom.scss          # Tema visual del sitio
├── _quarto.yml          # Configuración del proyecto Quarto
└── docs/                # Sitio renderizado (lo publica GitHub Pages)
```

Las diapositivas usan [reveal.js](https://quarto.org/docs/presentations/revealjs/) con el tema [`clean`](https://github.com/grantmcdermott/quarto-revealjs-clean); las páginas del sitio usan el formato HTML de Quarto con un tema propio (`custom.scss`).

## Compilar el sitio localmente

Necesitan [Quarto](https://quarto.org/) (incluido en RStudio) y R. Desde la raíz del repositorio:

```bash
quarto render
```

El sitio se genera en `docs/`, que GitHub Pages publica automáticamente. Para previsualizar con recarga en vivo:

```bash
quarto preview
```

## Requisitos de software

La lista completa de paquetes de R, junto con la instalación de Ollama y la descarga del modelo local, está en la sección [Software necesario](https://danilofreire.github.io/introduccion-ia-ucu/programa.html#software-necesario) del programa.

## Licencia

El contenido se distribuye bajo licencia [MIT](https://opensource.org/licenses/MIT).

## Contacto

Danilo Freire, <danilofreire@gmail.com>
