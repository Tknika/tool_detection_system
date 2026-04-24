# Kortxovision

## Dataset

El dataset está disponible en HuggingFace: [mikeldiez/kortxovision_dataset](https://huggingface.co/datasets/mikeldiez/kortxovision_dataset)

Para cargar el dataset:

```python
from datasets import load_dataset

ds = load_dataset("mikeldiez/kortxovision_dataset", trust_remote_code=True)
```

El dataset incluye los splits `test` y `validation`.

## Notebooks

Los notebooks están en la carpeta `notebooks/` y están diseñados para ejecutarse en Google Colab:

1. **`1_explore_ds.ipynb`**: Explora y visualiza el dataset de kortxovision
2. **`2_train_example.ipynb`**: Ejemplo de entrenamiento con el dataset
3. **`3_inference_example.ipynb`**: Ejemplo de inferencia con modelos entrenados

Para ejecutar en Colab:
```bash
!git clone https://github.com/TKNIKA/kortxovision.git
%cd kortxovision
```

## Inferencia

Para realizar inferencia con los modelos entrenados, consulta el notebook `3_inference_example.ipynb`.

Ejemplo básico:

```python
# Cargar modelo entrenado
# Realizar predicciones sobre nuevas imágenes
# Ver resultados
```

Las imágenes de prueba están disponibles en la carpeta `test_imgs/`.


## Entrenamiento usando gpu_docker
para realizar este paso es necesario tener instalado docker y nvidia-docker
[https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)  

1. Clonar el repositorio 
```bash
git clone https://github.com/TKNIKA/kortxovision.git
```
2. Entrar a la carpeta del repositorio
```bash
bash build.sh
```

3. lanzar en entrnamiento de manera normal
```bash
./train.sh  train_yolo.py kortxovision_dataset/ kortxovision_dataset/data.yaml env.example
```


4. Lanzar con busqueda de hiperparametros
```bash
./train.sh --detach train_yolo_optuna.py kortxovision_dataset/ kortxovision_dataset/data.yaml env.optuna.example 
```

5. Lazar tensorboard
```bash
bash tensorboard.sh 
```