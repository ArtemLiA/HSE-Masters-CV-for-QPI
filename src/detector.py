import numpy as np
import cv2


class GaussianBlurTransformer:
    def __init__(self, ksize, sigmaX, sigmaY=0):
        self.params = {"ksize": ksize, "sigmaX": sigmaX, "sigmaY": sigmaY}

    def transform(self, image: np.ndarray):
        image= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.GaussianBlur(image, **self.params)
        return image


class HoughCirclesDetector:
    def __init__(self, dp, param1, param2, minDistRate, minRadius, maxRadius, blur_params):
        self.hough_params = {
            "method": cv2.HOUGH_GRADIENT,
            "dp": dp, "param1": param1, "param2": param2,
            "minRadius": minRadius, "maxRadius": maxRadius,
        }
        self.minDistRate = minDistRate
        self.transformer = GaussianBlurTransformer(**blur_params)

    def detect(self, image: np.ndarray):
        """
        Обнаружить окружности на изображении

        Args:
            image (np.ndarray): изображение в формате RGB.
                Размер: [height, width, 3]
    
        """
        image = self.transformer.transform(image)
        circles = cv2.HoughCircles(
            image=image,
            minDist=self.minDistRate * np.max(image.shape),
            **self.hough_params,
        )
        return circles[0] if circles is not None else None
