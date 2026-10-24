import numpy as np


def solve(A):
    # TODO: Return the Frobenius norm of A.
    A = np.array(A,dtype=float)
    return float(np.linalg.norm(A,ord="fro"))
    pass
