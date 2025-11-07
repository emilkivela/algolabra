from src.services.formulas import power_iteration, rayleigh_quotient
import numpy as np

def test_power_iteration_finds_eigenvector_of_biggest_eigenvalue():
    test_matrix = np.array([[2,0],[0,1]])
    eigvals, eigvectors = np.linalg.eigh(test_matrix)
    correct = eigvectors[:,-1]
    result = power_iteration(test_matrix, 50)
    if np.dot(correct, result) < 0:
        result = -result

    assert np.allclose(result, correct, atol=1e-02)

def test_rayleigh_quotient_finds_eigenvalue_of_eigenvector():
    test_matrix = np.array([[2,0],[0,1]])
    eigvals, eigvectors = np.linalg.eigh(test_matrix)
    correct = eigvals[-1]
    assert np.isclose(rayleigh_quotient(eigvectors[:,-1], test_matrix), correct, atol=1e-04)


