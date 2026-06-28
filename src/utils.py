import numpy as np
import matplotlib.pyplot as plt


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

