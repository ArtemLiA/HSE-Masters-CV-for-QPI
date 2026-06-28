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
    file_path = Path(annotation_fname)
    file_dir = file_path.parent

    image_path = file_dir / "images" / f"{image_id}.png"
    image = cv2.imread(image_path)
    
    if image is None:
        return None
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_json(fname, *args, **kwargs):
    with open(fname, encoding="utf-8") as file:
        return json.load(file, *args, **kwargs)


def save_json(data, fname, *args, **kwargs):
    with open(fname, "w", encoding="utf-8") as file:
        return json.dump(data, file, *args, **kwargs)


def load_dataset(fname):
    """
    Прочитать датасет
    """
    annotation = read_json(fname)
    result = []

    for image_id, image_annotation in annotation.items():
        ellipses = [
            Ellipse(label="data", **obj)
            for obj in image_annotation["Эритроцит"]
        ]
        
        result.append(
            ImageAnnotation(
                image_id=image_id, ellipses=ellipses, image=read_image(fname, image_id)
            ))
    return result

