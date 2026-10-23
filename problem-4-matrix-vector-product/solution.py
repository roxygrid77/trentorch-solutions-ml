import numpy as np


def solve(a, b):
    """Implement Matrix-Vector Product from the mathematical contract in README.md."""
    a = np.array(a,float)
    b = np.array(b,float)
    return np.array([row @ b for row in a])
