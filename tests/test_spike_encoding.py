import numpy as np
import torch

from event_aug.spike_encoding import (
    delta_intensity_code_arr,
    delta_intensity_code_video,
    rate_code,
)


def test_rate_code():

    data_n = np.array([[1, 2], [3, 4]])
    data_t = torch.from_numpy(data_n)

    out = rate_code(data_n)
    assert out.shape == (2, 2)

    out = rate_code(data_t, n_steps=2)
    assert out.shape == (2, 2, 2)

    data = np.stack([data_n, data_n], axis=0)
    out = rate_code(data, gain=0.1)
    assert out.shape == (2, 2, 2)

    data = torch.stack([data_t, data_t], axis=0)
    out = rate_code(data, n_steps=2, gain=0.3)
    assert out.shape == (2, 2, 2)


def test_delta_intensity_code_arr():

    data = (np.random.rand(10, 32, 32) * 255).astype(np.uint8)

    out = delta_intensity_code_arr(data, threshold=25)
    assert out.shape == (10, 32, 32)

    out = delta_intensity_code_arr(data, percent_threshold=10, mode="percent_threshold")
    assert out.shape == (10, 32, 32)

    out = delta_intensity_code_arr(data, threshold=25, use_neg_delta=False)
    assert out.shape == (10, 32, 32)

    out = delta_intensity_code_arr(data, threshold=25, exclude_start=True)
    assert out.shape == (9, 32, 32)

    out = delta_intensity_code_arr(
        torch.from_numpy(data), threshold=25, exclude_start=True, use_neg_delta=False
    )
    assert out.shape == (9, 32, 32)

    data = (np.random.rand(10, 32, 32, 3) * 255).astype(np.uint8)

    out = delta_intensity_code_arr(data, threshold=25)
    assert out.shape == (10, 32, 32)


def test_delta_intensity_code():

    data = (np.random.rand(10, 32, 32) * 255).astype(np.uint8)

    out = delta_intensity_code_video(
        video_arr=data, threshold=25, use_neg_delta=False, exclude_start=False
    )
    assert out.shape == (10, 32, 32)
