from typing import Tuple

import cv2
import numpy as np
from noise import pnoise2
from perlin_numpy import generate_fractal_noise_3d, generate_perlin_noise_3d

from event_aug.utils import imgs_to_video


def shift_range(arr: np.ndarray):

    """
    Shift the range of values in an array to [0, 255]

    Parameters
    ----------
    arr: np.ndarray
        Input array

    Returns
    -------
    np.ndarray
        Array with shifted range
    """

    arr -= np.min(arr)
    arr /= np.max(arr)
    arr = arr * 255

    return arr


def downsample(
    arr: np.ndarray, reshape_size: Tuple[int, int], crop_size: Tuple[int, int] = None
) -> np.ndarray:

    """
    Downsample an array along the spatial dimensions to a given size by either resizing or cropping or both.

    Parameters
    ----------
    arr: np.ndarray
        Input array.
    reshape_size: Tuple[int, int]
        Size to reshape the array along spatial dimensions to using interpolation.
    crop_size: Tuple[int, int]
        Size to crop the array along spatial dimensions to.

    Returns
    -------
    np.ndarray
        Downsampled array.
    """

    if reshape_size is not None:

        assert len(arr.shape) in (
            2,
            3,
            4,
        ), "Input array must be a single-channel / multi-channel image or video."

        if len(arr.shape) == 2:
            arr = cv2.resize(arr, reshape_size)

        elif len(arr.shape) == 3 and arr.shape[2] in (1, 3):
            arr = cv2.resize(arr, reshape_size)

        else:
            resized_arrs = []
            for i in range(arr.shape[0]):
                resized_arrs.append(cv2.resize(arr[i], reshape_size))

            arr = np.array(resized_arrs)

    if crop_size is not None:
        arr = arr[..., : crop_size[0], : crop_size[1]]

    return arr


def postprocess_3d_noise(arr, reshape_size=None, crop_size=None):

    arr = downsample(arr, reshape_size, crop_size)
    arr = shift_range(arr)

    return arr


def gen_perlin_2d(
    shape: Tuple[int, int],
    scale: int = 100,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
    save_path: str = None,
    reshape_size: Tuple[int, int] = None,
    crop_size: Tuple[int, int] = None,
) -> np.ndarray:

    """
    Generates 2D Perlin noise.

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
    save_path: str
        Path to save the generated noise as an image.
    reshape_size: Tuple[int, int]
        Size to reshape the array along spatial dimensions to using interpolation.
    crop_size: Tuple[int, int]
        Size to crop the array along spatial dimensions to.

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

    arr = downsample(arr, reshape_size, crop_size)

    return arr


def gen_perlin_3d(
    shape: Tuple[int, int, int],
    res: Tuple[int, int, int] = (1, 4, 4),
    tileable: Tuple[bool, bool, bool] = (True, False, False),
    reshape_size: Tuple[int, int] = None,
    crop_size: Tuple[int, int] = None,
    save_path: str = None,
    out_fps: int = 25,
) -> np.ndarray:

    """
    Generates 3D Perlin noise.

    Parameters
    ----------
    shape: Tuple[int, int, int]
        Shape of the 3D Perlin noise wanted. This must be a multiple of res.
    res: Tuple[int, int, int]
        The number of periods of noise to generate along each
        axis (tuple of three ints). Note that shape must be a multiple of res.
    tileable: Tuple[bool, bool, bool]
        If the noise should be tileable along each axis.
    reshape_size: Tuple[int, int]
        Size to reshape the array along spatial dimensions to using interpolation.
    crop_size: Tuple[int, int]
        Size to crop the array along spatial dimensions to.
    save_path: str
        Path (.mp4) to save the generated 3D noise as a video.

    Returns
    -------
    np.ndarray
        3D array of Perlin noise.
    """

    noise = generate_perlin_noise_3d(shape, res, tileable)
    noise = postprocess_3d_noise(noise, reshape_size, crop_size)

    if save_path is not None:
        imgs_to_video(save_path, img_arr=noise, out_fps=out_fps)

    return noise


def gen_fractal_3d(
    shape: Tuple[int, int, int],
    res: Tuple[int, int, int] = (1, 4, 4),
    tileable: Tuple[bool, bool, bool] = (True, False, False),
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: int = 2,
    reshape_size: Tuple[int, int] = None,
    crop_size: Tuple[int, int] = None,
    save_path: str = None,
    out_fps: int = 25,
) -> np.ndarray:

    """
    Generates 3D Fractal noise.

    Parameters
    ----------
    shape: Tuple[int, int, int]
        Shape of the 3D fractal noise wanted. This must be a multiple of lacunarity**(octaves-1)*res.
    res: Tuple[int, int, int]
        The number of periods of noise to generate along each
        axis (tuple of three ints). Note that shape must be a multiple of
        (lacunarity**(octaves-1)*res).
    tileable: Tuple[bool, bool, bool]
        If the noise should be tileable along each axis.
    octaves: int
        Number of octaves in the noise.
    persistence: float
        The scaling factor between two octaves.
    lacunarity: int
            The frequency factor between two octaves.
    reshape_size: Tuple[int, int]
        Size to reshape the array along spatial dimensions to using interpolation.
    crop_size: Tuple[int, int]
        Size to crop the array along spatial dimensions to.
    save_path: str
        Path (.mp4) to save the generated 3D noise as a video.

    Returns
    -------
    np.ndarray
        3D array of Fractal noise.
    """

    noise = generate_fractal_noise_3d(
        shape, res, octaves, persistence, lacunarity, tileable
    )
    noise = postprocess_3d_noise(noise, reshape_size, crop_size)

    if save_path is not None:
        imgs_to_video(save_path, img_arr=noise, out_fps=out_fps)

    return noise
