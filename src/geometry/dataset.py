from typing import Optional
from dataclasses import dataclass
from pathlib import Path

import json

import numpy as np
import cv2

from .core import Ellipse


@dataclass
class ImageAnnotation:
    image_id: str
    ellipses: list[Ellipse]
    image: Optional[np.ndarray] = None


def read_image(annotation_fname, image_id):
    """
    Прочитать изображение

    Args:
        annotation_fname (str): путь к файлу с JSON-файлу с разметкой
            Файл с изображением должен лежать в папке images в той же 
            директории, что и JSON-файл
        image_id (str): идентификатор изображения

    Returns:
        np.ndarray: изображение
    """
    file_path = Path(annotation_fname)
    file_dir = file_path.parent

    image_path = file_dir / "images" / f"{image_id}.png"
    image = cv2.imread(image_path)
    
    if image is None:
        return None
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_json(fname, *args, **kwargs):
    """
    Прочитать JSON-файл
    """
    with open(fname, encoding="utf-8") as file:
        return json.load(file, *args, **kwargs)


def save_json(data, fname, *args, **kwargs):
    """
    Сохранить данные в JSON-файл
    """
    with open(fname, "w", encoding="utf-8") as file:
        return json.dump(data, file, *args, **kwargs)


def load_dataset(fname, label_name_mapping=None):
    """
    Загрузить данные

    Args:
        fname (str): путь к файлу с JSON-файлу с разметкой.
            Предполагается, что изображения лежат в папке images в той
            же директории, что и JSON-файл
        label_name_mapping (dict[str, str]): словарь для переименования классов
            (по умолчанию None)
    
    Returns
        list[ImageAnnotation]: список объектов ImageAnnotation

    """
    if label_name_mapping is None:
        label_name_mapping = {}

    annotation = read_json(fname)
    result = []

    for image_id, image_annotation in annotation.items():
        ellipses = []
        
        for class_name, objects in image_annotation.items():
            label_name = label_name_mapping.get(class_name, class_name)

            ellipses.extend([
                Ellipse(label=label_name, **obj)
                for obj in objects
            ])
        
        result.append(
            ImageAnnotation(
                image_id=image_id, ellipses=ellipses, image=read_image(fname, image_id)
            ))
    return result
