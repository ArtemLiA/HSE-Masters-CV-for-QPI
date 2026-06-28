from typing import List

import numpy as np

from .core import Ellipse


def intersection_over_union(e1: Ellipse, e2: Ellipse):
    """
    Вычислить Intersection over Union (IoU) для двух эллипсов
    """
    p1 = e1.polygon
    p2 = e2.polygon

    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter / union


def mean_intersection_over_union(
    ellipses_true: List[Ellipse], ellipses_pred: List[Ellipse]
):
    """
    Посчитать среднее IoU для истинных и предсказанных эллипсов
    """
    assert len(ellipses_true) == len(ellipses_pred)

    return np.mean([
        intersection_over_union(e1, e2)
        for e1, e2 in zip(ellipses_true, ellipses_pred)
    ])


def std_intersection_over_union(
    ellipses_true: List[Ellipse], ellipses_pred: List[Ellipse]
):
    """
    Посчитать стандартное отклонение IoU для истинных и предсказанных эллипсов
    """
    assert len(ellipses_true) == len(ellipses_pred)

    return np.std([
        intersection_over_union(e1, e2)
        for e1, e2 in zip(ellipses_true, ellipses_pred)
    ])
