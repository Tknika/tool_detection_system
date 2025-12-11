#!/usr/bin/env python3
"""
Script para juntar múltiples datasets YOLO en uno nuevo.
Renombra todas las imágenes y labels desde 00000.
Se queda con el data.yml que tenga más clases.
"""

import os
import sys
import shutil
import re
from pathlib import Path
from dotenv import load_dotenv
import yaml
import logging
from collections import defaultdict

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_yaml_classes(yaml_path):
    """Carga las clases de un archivo data.yml"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        return data.get('names', [])


def normalize_path(path_str):
    """
    Normaliza una ruta: expande ~, variables de entorno.
    """
    if not path_str:
        return None
    
    # Limpiar espacios en blanco
    path_str = path_str.strip()
    
    # Expandir variables de entorno (ej: $HOME, ${HOME})
    path_str = os.path.expandvars(path_str)
    
    # Expandir ~
    path_str = os.path.expanduser(path_str)
    
    # Crear Path
    path = Path(path_str)
    
    # VALIDAR: SOLO aceptar rutas absolutas
    if not path.is_absolute():
        logger.error(f"Error: Ruta relativa no permitida: {path_str}")
        logger.error("Todas las rutas deben ser absolutas (empezar con /)")
        return None
    
    # Resolver symlinks y . ..
    path = path.resolve()
    
    return path


def get_all_datasets_from_env():
    """
    Busca todos los datasets en el .env siguiendo el patrón:
    dataset_N_images_path, dataset_N_labels_path, dataset_N_data_yml_path
    Retorna un diccionario: {N: {'images': Path, 'labels': Path, 'yaml': Path}}
    """
    datasets = defaultdict(dict)
    env_vars = dict(os.environ)
    
    # Patrón para buscar dataset_N_*
    pattern = re.compile(r'^dataset_(\d+)_(images_path|labels_path|data_yml_path)$')
    
    for key, value in env_vars.items():
        match = pattern.match(key)
        if match:
            num = int(match.group(1))
            field = match.group(2)
            
            # Normalizar la ruta
            normalized_path = normalize_path(value)
            if normalized_path is None:
                logger.error(f"ERROR: {key} tiene una ruta inválida: {value}")
                logger.error(f"Todas las rutas deben ser absolutas (empezar con /)")
                logger.error(f"Ejemplo: /home/mikel/github/TKNIKA/kortxovision/datasets_jon_ander/...")
                continue
            
            if field == 'images_path':
                datasets[num]['images'] = normalized_path
            elif field == 'labels_path':
                datasets[num]['labels'] = normalized_path
            elif field == 'data_yml_path':
                datasets[num]['yaml'] = normalized_path
    
    # Validar que cada dataset tenga los 3 campos
    valid_datasets = {}
    for num, dataset in datasets.items():
        if 'images' in dataset and 'labels' in dataset and 'yaml' in dataset:
            valid_datasets[num] = dataset
        else:
            logger.warning(f"Dataset {num} incompleto, saltando...")
            missing = []
            if 'images' not in dataset:
                missing.append('images')
            if 'labels' not in dataset:
                missing.append('labels')
            if 'yaml' not in dataset:
                missing.append('yaml')
            logger.warning(f"  Faltan: {', '.join(missing)}")
    
    return valid_datasets


def get_yaml_with_more_classes(all_yaml_paths):
    """Retorna el path del yaml con más clases de todos"""
    best_path = None
    max_classes = -1
    
    for yaml_path in all_yaml_paths:
        classes = load_yaml_classes(yaml_path)
        num_classes = len(classes)
        if num_classes > max_classes:
            max_classes = num_classes
            best_path = yaml_path
    
    return best_path, max_classes


def merge_datasets():
    """Función principal para juntar los datasets"""
    
    # Cargar output path
    output_path_str = os.getenv('output_dataset_path')
    if not output_path_str:
        logger.error("Error: output_dataset_path no está definido en .env")
        return False
    
    output_path = normalize_path(output_path_str)
    if output_path is None:
        logger.error(f"Error: output_dataset_path no es válido: {output_path_str}")
        return False
    
    # Buscar todos los datasets en el .env
    datasets = get_all_datasets_from_env()
    
    if len(datasets) == 0:
        logger.error("Error: No se encontraron datasets en el .env")
        logger.error("Formato esperado: dataset_N_images_path, dataset_N_labels_path, dataset_N_data_yml_path")
        logger.error("Buscando variables que empiecen con 'dataset_'...")
        env_vars = [k for k in os.environ.keys() if k.startswith('dataset_')]
        if env_vars:
            logger.error(f"Variables encontradas: {', '.join(sorted(env_vars))}")
        return False
    
    logger.info(f"Encontrados {len(datasets)} dataset(s) para mergear")
    
    # Validar que existan todos los paths
    for num, dataset in sorted(datasets.items()):
        logger.info(f"Validando dataset {num}...")
        for field, path in dataset.items():
            logger.info(f"  {field}: {path}")
            
            # Para archivos yaml, intentar con .yml y .yaml si no existe
            if field == 'yaml' and not path.exists():
                # Intentar con la extensión alternativa
                if path.suffix == '.yml':
                    alt_path = path.with_suffix('.yaml')
                elif path.suffix == '.yaml':
                    alt_path = path.with_suffix('.yml')
                else:
                    alt_path = None
                
                if alt_path and alt_path.exists():
                    logger.info(f"  Archivo {path.name} no existe, usando {alt_path.name} en su lugar")
                    dataset['yaml'] = alt_path
                    path = alt_path
                else:
                    logger.error(f"Error: dataset_{num}_{field} no existe: {path}")
                    if alt_path:
                        logger.error(f"  Tampoco existe la alternativa: {alt_path}")
                    logger.error(f"  Path absoluto: {path.absolute()}")
                    return False
            
            if not path.exists():
                logger.error(f"Error: dataset_{num}_{field} no existe: {path}")
                logger.error(f"  Path absoluto: {path.absolute()}")
                logger.error(f"  ¿Es un directorio? {path.is_dir()}")
                logger.error(f"  ¿Es un archivo? {path.is_file()}")
                return False
            else:
                logger.info(f"  ✓ {field} existe")
    
    # Recopilar todos los yaml para elegir el mejor
    all_yaml_paths = [dataset['yaml'] for dataset in datasets.values()]
    best_yaml_path, num_classes = get_yaml_with_more_classes(all_yaml_paths)
    logger.info(f"Usando data.yml con {num_classes} clases: {best_yaml_path}")
    
    # Crear estructura de directorios de salida
    output_images = output_path / 'images'
    output_labels = output_path / 'labels'
    
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creando dataset en: {output_path}")
    
    # Obtener todas las imágenes de todos los datasets
    all_images = []
    
    for num, dataset in sorted(datasets.items()):
        images_path = dataset['images']
        labels_path = dataset['labels']
        
        logger.info(f"Procesando dataset {num}...")
        
        # Contar todas las imágenes encontradas
        all_image_files = [p for p in sorted(images_path.glob('*')) 
                          if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        logger.info(f"  Total de imágenes encontradas en dataset {num}: {len(all_image_files)}")
        
        images_with_labels = 0
        images_without_labels = 0
        
        for img_path in all_image_files:
            label_path = labels_path / f"{img_path.stem}.txt"
            if label_path.exists():
                all_images.append((img_path, label_path))
                images_with_labels += 1
            else:
                images_without_labels += 1
                if images_without_labels <= 10:  # Solo mostrar primeros 10
                    logger.warning(f"  Label no encontrado para {img_path.name}, saltando...")
        
        if images_without_labels > 10:
            logger.warning(f"  ... y {images_without_labels - 10} imágenes más sin labels")
        
        logger.info(f"  Dataset {num}: {images_with_labels} con labels, {images_without_labels} sin labels")
    
    logger.info(f"Total de imágenes a copiar (con labels): {len(all_images)}")
    
    # Copiar y renombrar desde 000000
    for idx, (img_path, label_path) in enumerate(all_images):
        # Formatear número con ceros a la izquierda (7 dígitos)
        new_name = f"{idx:08d}"
        
        # Mantener la extensión original de la imagen
        new_img_name = f"{new_name}{img_path.suffix}"
        new_label_name = f"{new_name}.txt"
        
        new_img_path = output_images / new_img_name
        new_label_path = output_labels / new_label_name
        
        # Copiar imagen
        shutil.copy2(img_path, new_img_path)
        
        # Copiar label
        shutil.copy2(label_path, new_label_path)
        
        if (idx + 1) % 100 == 0:
            logger.info(f"Procesadas {idx + 1}/{len(all_images)} imágenes...")
    
    # Copiar el data.yml elegido
    output_yaml = output_path / 'data.yml'
    shutil.copy2(best_yaml_path, output_yaml)
    
    logger.info(f"✓ Merge completado: {len(all_images)} imágenes copiadas a {output_path}")
    logger.info(f"✓ data.yml copiado desde {best_yaml_path}")
    
    return True


if __name__ == '__main__':
    success = merge_datasets()
    if not success:
        sys.exit(1)

