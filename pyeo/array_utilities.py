"""
pyeo.array_utilities
===============
Contains routines for manipulating arrays.
"""

import numpy as np


def project_array(array_in, depth, axis):
    """Returns a new array with an extra dimension. Data is projected along that dimension to depth."""
    array_in = np.expand_dims(array_in, axis)
    array_in = np.repeat(array_in, depth, axis)
    return array_in