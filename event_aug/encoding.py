from typing import Union

import numpy as np
import torch


def rate_code(
    arr: Union[np.ndarray, torch.Tensor], n_steps: int = 1, gain: int = 1
) -> np.ndarray:

    """
    Converts array of continuous input data to spikes using rate coding.
    Input array is normalized to [0, 1] and the resulting elements are treated as
    probabilities for Bernoulli trials.

    Parameters
    ----------
    arr: np.ndarray or torch.Tensor
        2D array of continuous input data. 3D array in case of time-series data.
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

    arr = arr / np.linalg.norm(arr)
    arr = arr * gain

    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr).float()

    assert (
        len(arr.shape) == 2 or len(arr.shape) == 3
    ), "Input must be a 2D array of probabilities or a 3D array in case of time series data"

    if len(arr.shape) == 2 and n_steps > 1:
        new_arr = torch.clone(arr)
        for _ in range(n_steps - 1):
            new_arr = torch.stack((new_arr, arr))
        arr = new_arr

    spikes = torch.bernoulli(arr)

    return spikes.numpy()
