import numpy as np

from event_aug.encoding import rate_code


def test_rate_code():

    arr = np.array([[1, 2], [3, 4]])
    out = rate_code(arr)
    assert out.shape == (2, 2)

    arr = np.array([[1, 2], [3, 4]])
    out = rate_code(arr, n_steps=2)
    assert out.shape == (2, 2, 2)

    arr = np.stack([arr, arr], axis=0)
    out = rate_code(arr)
    assert out.shape == (2, 2, 2)
