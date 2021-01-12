import numpy as np
from scipy.special import sph_harm

from matplotlib import pyplot as plt


def generate_input_channels(harmonics=4, ENVMAP_SZ=(16, 32)):
    theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, ENVMAP_SZ[1]),
                             np.linspace(0, np.pi, ENVMAP_SZ[0]))

    x, y = np.meshgrid(np.linspace(0, 1, ENVMAP_SZ[1]), np.linspace(0, 1, ENVMAP_SZ[0]))
    retval = [x, 1 - x, y]

    for l in range(1, harmonics):
        for m in range(l + 1):
            grid = sph_harm(m, l, theta, phi).real
            grid = grid - grid.min()
            grid = grid / grid.max()
            retval.append(grid)
            retval.append(1 - grid)
            if m != 0:
                grid = sph_harm(m, l, theta, phi).imag
                grid = grid - grid.min()
                grid = grid / grid.max()
                retval.append(grid)
                retval.append(1 - grid)

    retval = np.asarray(retval)
    return retval


if __name__ == '__main__':
    data = generate_input_channels(harmonics=3)
    print(data.shape)
    for i, channel in enumerate(range(data.shape[0])):
        plt.subplot(5, 5, i + 1); plt.imshow(data[i,...])
    plt.show()
