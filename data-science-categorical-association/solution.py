import numpy as np


def contingency_table(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    label_a = sorted(set(a.tolist()))
    label_b = sorted(set(b.tolist()))
    row_index = {label:i for i,label in enumerate(label_a)}
    col_index = {label:j for j,label in enumerate(label_b)}
    table = np.zeros((len(label_a),len(label_b)),dtype = int)
    for x,y in zip(a,b):
        i = row_index[x]
        j = col_index[y]
        table[i,j] += 1
    return table

def expected_counts(table: np.ndarray) -> np.ndarray:
    table = np.asarray(table,dtype=float)
    row_total = table.sum(axis=1)
    col_total = table.sum(axis=0)
    grand_total = table.sum()
    if grand_total == 0:
        return np.zeros(table.shape,dtype=float)
    return np.outer(row_total,col_total)/grand_total


def chi_square_statistic(table: np.ndarray) -> float:
    observed = np.asarray(table,dtype=float) 
    expected = expected_counts(observed)
    valid = expected > 0
    difference = observed[valid] - expected[valid]
    return float(np.sum(difference ** 2/expected[valid]))


def cramers_v(table: np.ndarray) -> float:
    table = np.asarray(table,dtype = float)
    r,c = table.shape
    n = table.sum()
    deno = min(r-1,c-1)
    if deno <= 0 or n == 0:
        return 0.0
    chi = chi_square_statistic(table)
    return float(np.sqrt(chi/(n*deno)))
