
import numpy as np

from _load import load_solution

correlation = load_solution("math-expectation-variance-covariance").correlation


def correlation_matrix(x: np.ndarray) -> np.ndarray:
    num_features = x.shape[1]
    result = np.eye(num_features)
    for i in range(num_features):
        for j in range(i+1,num_features):
            corr = correlation(x[:,i],x[:,j])
            result[i,j] = corr
            result[j,i] = corr
    return result
    


def most_correlated_pair(corr_matrix: np.ndarray) -> tuple[int, int]:
    matrix = np.abs(corr_matrix.copy())
    np.fill_diagonal(matrix,-np.inf)
    i,j = np.unravel_index(np.argmax(matrix),matrix.shape)
    return (int(i),int(j))
