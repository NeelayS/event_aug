import numpy as np

from event_aug.encoding import rate_code


def test_rate_code():
    arr = np.array([[1, 2], [3, 4]])
    _ = rate_code(arr)
