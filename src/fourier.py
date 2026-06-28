import numpy as np
from scipy.ndimage import maximum_position
from skimage.restoration import unwrap_phase


def image_to_grey(image: np.ndarray):
  """
  Перевести изображение в 2D-массив (чёрно-белое)

  Args:
    image (np.ndarray): трёхканальное изображение
        Массив формы [h, w, 3]

  """
  assert image.ndim == 3
  return np.dot(image[..., :3], [0.299, 0.587, 0.114])


def first_order_position(F: np.ndarray, rel_radius: float = 0.05):
    """
    Получить позицию (координаты), соответствующие 1-ой частоте
    
    Args:
        F (np.ndarray): частнотное поле
        rel_radius (float): относительный (к размеру изображения)
            радиус DC (максимума 0 порядка), который будет проигнорирован
            при поиске максимума

    """
    F = F.copy()
    
    n_rows, n_cols = F.shape
    y_center, x_center = n_rows // 2, n_cols // 2

    abs_radius = int(min(n_rows, n_cols) * rel_radius)
    ys, xs = np.mgrid[:n_rows, :n_cols]

    mask = ((xs - x_center) ** 2 + (ys - y_center) ** 2) <= abs_radius ** 2
    F[mask] = 0.0

    max_pos = maximum_position(F)
    return max_pos


def move_position(F, position):
    """
    Осуществить сдвиг изображения так, что пиксель с координатами
    position оказался в середине изображения
    
    Args:
        F (np.ndarray): частотное после
        position (np.ndarray, list): координаты нового центра в 
            формате (y_peak, x_peak) 

    """
    F = F.copy()
    n_rows, n_cols = F.shape
  
    y_center, x_center = n_rows // 2, n_cols // 2
    y_peak, x_peak = position

    dy = y_center - y_peak
    dx = x_center - x_peak

    F = np.roll(F, (dy, dx), axis=(0, 1))
    return F


def remove_carrier(F, rel_radius=0.05):
    """
    Удалить все частоты вне центра изображения
    
    Args:
        F (np.ndarray): частотное поле
        rel_radius (float): относительный (размеру изображения)
            радиус центра, который будет оставлен

    """
    F = F.copy()

    n_rows, n_cols = F.shape
    y_center, x_center = n_rows // 2, n_cols // 2

    abs_radius = int(min(n_rows, n_cols) * rel_radius)
    ys, xs = np.mgrid[:n_rows, :n_cols]

    mask = ((xs - x_center) ** 2 + (ys - y_center) ** 2) > (abs_radius ** 2)
    F[mask] = 0.0
    return F


def double_fourier(image, fo_rel_radius=0.05, rc_rel_radius=0.08):
    """
    Восстановление фазы при помощи двойного преобразования Фурье
    (double fourier transform)

    Args:
        image (np.ndarray): изображение (uint или float) 
            Массив формы [h, w, 3]
        fo_rel_radius (float): относитель
    
    Returns:
        A (np.ndarray), phi (np.ndarray): амплитуда и развёрнутая фаза
            координатного поля (после обратного преобразования Фурье)

    """
    image = image_to_grey(image)

    F = np.fft.fft2(image)
    F = np.fft.fftshift(F)

    peak = first_order_position(F, rel_radius=fo_rel_radius)
    F = move_position(F, peak)
    F = remove_carrier(F, rel_radius=rc_rel_radius)

    F_new = np.fft.ifftshift(F)
    F_new = np.fft.ifft2(F_new)

    A_new, phi_new = np.abs(F_new), np.angle(F_new)
    phi_new = unwrap_phase(phi_new)
    return A_new, phi_new

