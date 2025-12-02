from src.services.eigenfaces import get_eigs, calculate_eigenfaces
import numpy as np

def test_get_eigs_finds_correct_values():
    test_matrix = np.array([[2,0],[0,1]])
    eigvals, eigvecs = np.linalg.eigh(test_matrix)
    i = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[i]
    eigvecs_sorted = eigvecs[:, i]
    test_eigvals, test_eigvectors = get_eigs(test_matrix, 2)
    test_eigvectors = np.column_stack(test_eigvectors)
    assert np.allclose(test_eigvectors, eigvecs_sorted, atol=1e-02)
    assert np.allclose(test_eigvals, eigvals_sorted)

def test_calculate_eigenfaces_returns_correct_shapes():
    
    return




