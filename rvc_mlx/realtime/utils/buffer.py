import numpy as np


def circular_write(new_data, target):
    """
    Circular write for numpy arrays.
    new_data: (L,)
    target: (Target_L,)
    """
    offset = new_data.shape[0]
    target[:-offset] = target[offset:]
    target[-offset:] = new_data
    return target
