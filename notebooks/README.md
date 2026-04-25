# Notebooks en Google Colab (guia basica)

Si, se puede incrustar un boton en GitHub para abrir cada notebook directamente en Colab.

## Abrir en Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tknika/tool_detection_system/blob/tutorial/notebooks/1_simple_train.ipynb) `1_simple_train.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tknika/tool_detection_system/blob/tutorial/notebooks/2_hp_search.ipynb) `2_hp_search.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tknika/tool_detection_system/blob/tutorial/notebooks/3_full_training.ipynb) `3_full_training.ipynb`

## Como correrlos (pasos rapidos)

1. Entra a este repo en GitHub.
2. Ve a la carpeta `notebooks/`.
3. Haz click en el boton **Open In Colab** del notebook que quieras.
4. En Colab, ejecuta las celdas de arriba hacia abajo (`Runtime -> Run all` o `Shift+Enter`).
5. Si Colab te pide permisos (Google Drive/Hugging Face), autorizalos cuando aparezca el prompt.

## Notas utiles

- Los notebooks ya tienen celdas para instalar dependencias (`ultralytics`, `huggingface_hub`, etc.).
- Si cambias rutas o nombres de archivos (por ejemplo `kortxo.jpg`), ajustalo en la celda de configuracion de inferencia.
- Para guardar resultados de entrenamiento (weights, runs), puedes montar Google Drive y usar una ruta dentro de `/content/drive/MyDrive/...`.

## Como generar un boton Open In Colab para cualquier notebook

Patron del enlace:

`https://colab.research.google.com/github/<owner>/<repo>/blob/<branch>/<path_al_notebook>.ipynb`

Ejemplo para este repo:

`https://colab.research.google.com/github/Tknika/tool_detection_system/blob/main/notebooks/3_full_training.ipynb`
