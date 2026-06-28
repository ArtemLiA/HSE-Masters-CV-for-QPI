import numpy as np
from .core import Ellipse


def opencv_hough_circles_to_ellipses(
    image: np.ndarray,
    opencv_hough_circles: np.ndarray | None
) -> list[Ellipse]:
    """
    Преобразовать массив из предсказаний от OpenCV HoughCircles
    в список эллипсов

    Args:
        image (np.ndarray): исходное изображение
        opencv_hough_circles (np.ndarray | None): массив из предсказаний от OpenCV HoughCircles

    """
    heigth, width = image.shape[:2]

    if opencv_hough_circles is None:
        return []

    return [
        Ellipse(x, y, r, r, 0.0, width=width, height=heigth, label="predict")
        for (x, y, r) in opencv_hough_circles
    ]
