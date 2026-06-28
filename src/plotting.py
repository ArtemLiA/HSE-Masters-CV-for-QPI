import numpy as np
import matplotlib.patches as patches

from .geometry import Ellipse


def draw_image(ax, image, off_axis=True):
    """
    Отрисовать изображение
    """
    if image.ndim == 2:
        image = np.expand_dims(image, -1)
        image = np.repeat(image, repeats=3, axis=2)

    if off_axis:
        ax.axis("off")
    
    ax.imshow(image)


def draw_circle(ax, x, y, r, **kwargs):
    """
    Отрисовать окружность с заданными параметрами
    """
    circle = patches.Circle(xy=(x, y), radius=r, **kwargs)
    ax.add_patch(circle)


def draw_ellipse(ax, x, y, rx, ry, angle=0, **kwargs):
    """
    Отрисовать эллипс с заданными параметрами
    """
    ellipse = patches.Ellipse(
        xy=(x, y), width=2 * rx, height=2 * ry, angle=angle, **kwargs
    )
    ax.add_patch(ellipse)


def draw_ellipse_geometry(ax, ellipse: Ellipse, **kwargs):
    """
    Отрисовать эллипс с заданными параметрами из словаря
    """
    x, y = ellipse.x, ellipse.y
    rx, ry = ellipse.rx, ellipse.ry
    angle = ellipse.angle

    draw_ellipse(ax, x, y, rx, ry, angle, **kwargs)


def draw_all_ellipses_geometry(ax, ellipses: list[Ellipse], **kwargs):
    """"
    Отрисовать все эллипсы из списка
    """
    for ellipse in ellipses:
        draw_ellipse_geometry(ax, ellipse, **kwargs)
