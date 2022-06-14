import numpy as np
import torch

from event_aug.encoding import rate_code


def test_rate_code():

    arr_n = np.array([[1, 2], [3, 4]])
    out = rate_code(arr_n)
    assert out.shape == (2, 2)

    arr_t = torch.Tensor([[1, 2], [3, 4]])
    out = rate_code(arr_t, n_steps=2)
    assert out.shape == (2, 2, 2)

    arr = np.stack([arr_n, arr_n], axis=0)
    out = rate_code(arr)
    assert out.shape == (2, 2, 2)

    arr = torch.stack([arr_t, arr_t], axis=0)
    out = rate_code(arr)
    assert out.shape == (2, 2, 2)
