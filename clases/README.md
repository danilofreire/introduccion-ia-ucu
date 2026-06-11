# Materiales de clase

Este directorio contiene las diapositivas, los laboratorios, las tareas y los datos del curso, organizados por día. Cada subcarpeta `dia-0X/` tiene su propio README con el detalle de las sesiones.

## Estructura por día

| Día | Tema | Carpeta |
|-----|------|---------|
| 1 | Fundamentos de IA y Machine Learning | [`dia-01/`](dia-01/) |
| 2 | Aprendizaje supervisado | [`dia-02/`](dia-02/) |
| 3 | Texto y aprendizaje no supervisado | [`dia-03/`](dia-03/) |
| 4 | Modelos de lenguaje (LLMs) | [`dia-04/`](dia-04/) |
| 5 | Modelos locales, ética y cierre | [`dia-05/`](dia-05/) |

Cada día combina dos sesiones teóricas con laboratorios prácticos en R, más una o dos tareas.

## Convención de archivos

Dentro de cada día encontrarán:

- `NN-*.qmd`: diapositivas de cada sesión, con numeración continua a lo largo del curso (01 a 20)
- `laboratorio-N.R`: el código de cada laboratorio en un script de R, para seguir la sesión sin copiar de las diapositivas
- `tarea-N.qmd` y `tarea-N-respuestas.qmd`: enunciado y clave de respuestas de cada tarea
- `datos/`: los conjuntos de datos que usan las sesiones y los laboratorios de ese día

Las diapositivas se escriben en [Quarto](https://quarto.org/) y se publican en el [sitio del curso](https://danilofreire.github.io/introduccion-ia-ucu). Para compilarlas localmente, ver el README en la raíz del repositorio.
