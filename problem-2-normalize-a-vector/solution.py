import numpy as np


def solve(x):
    """Implement Normalize a Vector from the mathematical contract in README.md."""
    x = np.array(x,dtype=float)
    n = np.linalg.norm(x)
    if n == 0:
        raise ValueError("ZERO VECTOR")
    return x/n
