import numpy as np

import matplotlib.pyplot as plt
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


def plot_amplitude(F, peak=None, show=True, **kwargs):
  """
  Визуализировать амплитуду
  """
  image = np.log(np.abs(F) + 1.0)

  plt.imshow(image, **kwargs)
  plt.colorbar()

  if peak is not None:
    idx_y, idx_x = peak
    plt.scatter([idx_x], [idx_y], color='red')

  if show:
    plt.show()


def plot_3d_phase(A, phi, pixel_size, wave_length, show=False, cmap='viridis'):
    """
    Визуализировать фазу в виде 3D-поверхности
    """
    n_rows, n_cols = A.shape

    Y, X = np.mgrid[:n_rows, :n_cols]
    X = pixel_size * X
    Y = pixel_size * Y
    Z = wave_length * phi / (2 * np.pi)
  
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=45, azim=45)
    surf = ax.plot_surface(X, Y, Z, label='Blood Cell', cmap=cmap)
    fig.colorbar(surf)
  
    ax.legend()
    ax.set_xlabel("длина (мкм)")
    ax.set_ylabel("ширина (мкм)")

    if show:
        plt.show()
