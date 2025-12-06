from src.services.eigenfaces import get_eigs, get_input_weight
from src.services.utils import load_dataset_faces
import numpy as np
from PIL import Image

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

def test_load_dataset_faces_returns_correct_shaped_matrix(tmp_path):
    testperson = tmp_path / "person1"
    testperson.mkdir()
    image1 = Image.fromarray(np.ones((5,5), dtype=np.uint8))
    image2 = Image.fromarray(np.full((5,5), 2, dtype=np.uint8))
    image1.save(testperson / "1.pgm")
    image2.save(testperson / "2.pgm")
    t_matrix = load_dataset_faces(tmp_path)
    assert t_matrix.shape == (25,2)

def test_get_input_weight_basic(monkeypatch):
    test_img = np.array([1, 2, 3, 4], dtype=np.float32)

    mean = np.zeros((4, 1), dtype=np.float32)

    centered = test_img - mean.flatten()

    eigfaces = np.array([
        [1, 0],   
        [0, 1],   
        [0, 0],   
        [0, 0],   
    ], dtype=np.float32)

    expected = np.dot(eigfaces.T, centered)

    def fake_load_input_face(_):
        return test_img, "Person 1"

    monkeypatch.setattr("src.services.eigenfaces.load_input_face", fake_load_input_face)

    result, label = get_input_weight("dummy_path", mean, eigfaces)

    assert label == "Person 1"
    assert np.allclose(result, expected)
