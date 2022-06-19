import numpy as np
import torch

from event_aug.spike_encoding import delta_intensity_code, rate_code


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


def test_delta_intensity_code():

    data_t = (torch.randn(10, 32, 32) * 255).type(torch.LongTensor)

    out = delta_intensity_code(data_t, threshold=25)
    assert out.shape == (10, 32, 32)

    out = delta_intensity_code(data_t, threshold=25, use_negative_delta=False)
    assert out.shape == (10, 32, 32)

    out = delta_intensity_code(data_t, threshold=25, ignore_start=True)
    assert out.shape == (9, 32, 32)

    out = delta_intensity_code(
        data_t.numpy(), threshold=25, ignore_start=True, use_negative_delta=False
    )
    assert out.shape == (9, 32, 32)
