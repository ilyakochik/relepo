import numpy as np


def iter_round(obj, decimals=2):
    """Recursively round iterable"""

    iter_obj = obj
    if isinstance(obj, list):
        iter_obj = enumerate(obj)
    elif isinstance(obj, dict):
        iter_obj = obj.items()
    elif isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, np.float32):
        return round(obj, decimals)
    elif isinstance(obj, np.float64):
        return round(obj, decimals)
    else:
        return obj

    for i, v in iter_obj:
        obj[i] = iter_round(obj[i], decimals)

    return obj
