from typing import Union

import numpy as np
import torch
from skvideo.io import vread, vwrite


def rate_code(
    data: Union[np.ndarray, torch.Tensor], n_steps: int = 1, gain: int = 1
) -> np.ndarray:

    """
    Converts array of continuous input data to spikes using rate coding.
    Input array is normalized to [0, 1] and the resulting elements are treated as
    probabilities for Bernoulli trials.

    Parameters
    ----------
    data: np.ndarray or torch.Tensor
        2D array of continuous input data. 3D array in case of time-series data (shape: T x H x W).
    n_steps: int
        Number of time-steps to perform rate coding for. Only used for 2D input data.
        In case of time series data, rate coding is performed for each time-step.
    gain: int
        Factor by which to scale normalized input features.

    Returns
    -------
    np.ndarray
        2D array of spikes. 3D array in case of time-series data.
    """

    data = data / np.linalg.norm(data)
    data = data * gain

    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data).float()

    assert (
        len(data.shape) == 2 or len(data.shape) == 3
    ), "Input must be a 2D array of probabilities or a 3D array in case of time series data"

    if len(data.shape) == 2 and n_steps > 1:
        new_data = torch.clone(data)
        for _ in range(n_steps - 1):
            new_data = torch.stack((new_data, data))
        data = new_data

    spikes = torch.bernoulli(data)

    return spikes.numpy()


def delta_intensity_code(
    data: Union[np.ndarray, torch.Tensor],
    threshold: int,
    use_negative_delta: bool = True,
    exclude_start: bool = False,
) -> Union[np.ndarray, torch.Tensor]:

    """
    Converts video data to spikes. If the difference in the intensity of a pixel in
    consecutive frames is greater than the threshold, the pixel is assigned an event.

    Parameters
    ----------
    data: np.ndarray or torch.Tensor
        Array containing the video data (frames). Should be of shape (T x H x W) or (T x H x W x C).
    threshold: int
        Threshold for the difference in intensities of pixels in consecutive frames
        for assigning an event.
    use_negative_delta: bool
        Whether to consider decreases in intensity as well along with increases.
    exclude_start: bool
        Whether to not return the spikes for the first frame which will always be 0 for all pixels.

    Returns
    -------
    np.ndarray or torch.Tensor
        Array containing the spikes.
    """

    assert (
        len(data.shape) == 3 or len(data.shape) == 4
    ), "Input must be an array with shape (T x H x W) or (T x H x W x C)"

    if isinstance(data, torch.Tensor):
        spikes = torch.zeros_like(data)
    else:
        spikes = np.zeros_like(data)

    for i in range(1, data.shape[0]):

        intensity_delta = data[i] - data[i - 1]
        if use_negative_delta:
            intensity_delta = abs(intensity_delta)

        spikes[i][intensity_delta > threshold] = 1

    if exclude_start:
        return spikes[1:]

    return spikes


def encode_video(
    video_path: str,
    save_out_video: bool = False,
    save_path: str = None,
    threshold: int = 25,
    use_negative_delta: bool = False,
    exclude_start: bool = False,
) -> np.ndarray:

    video = vread(video_path)
    spikes = delta_intensity_code(
        video,
        threshold=threshold,
        use_negative_delta=use_negative_delta,
        exclude_start=exclude_start,
    )

    if save_out_video is True:
        assert save_path is not None, "Path must be provided to save output video"

        spikes = (spikes * 255).astype(np.uint8)
        vwrite(save_path, spikes)

    return spikes
