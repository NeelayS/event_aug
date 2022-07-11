from typing import Tuple

import cv2
import numpy as np
from noise import pnoise2


def gen_perlin_2d(
    shape: Tuple[int, int] = (200, 200),
    scale: int = 100,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
    save_path: str = None,
) -> np.ndarray:

    """
    Generated 2D Perlin noise.

    Parameters
    ----------
    shape: Tuple[int, int]
        Shape of the 2D output array.
    scale: int
        Scale of the noise.
    octaves: int
        Number of octaves to use.
    persistence: float
        Scaling factor between two octaves.
    lacunarity: float
        Frequency factor between two octaves.
    seed: int
        Seed for the noise generation.

    Returns
    -------
    np.ndarray
        2D array of Perlin noise.
    """

    if not seed:
        seed = np.random.randint(0, 100)

    arr = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            arr[i][j] = pnoise2(
                i / scale,
                j / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=seed,
            )

    max_arr = np.max(arr)
    min_arr = np.min(arr)
    arr = (arr - min_arr) / (max_arr - min_arr)

    if save_path is not None:
        cv2.imwrite(save_path, arr * 255)

    return arr
