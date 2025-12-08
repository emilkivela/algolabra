from src.services.eigenfaces import (
    get_eigs, get_input_weight, calculate_eigenfaces,
      recognise_input_face, load_input_face)
from src.services.utils import load_dataset_faces
import numpy as np
from PIL import Image
import io
from werkzeug.datastructures import FileStorage

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

def test_calculate_eigenfaces_correct_shape_and_mean():
    T_matrix = np.array([
        [1,2,3],
        [2,3,4],
        [5,5,5]
    ])
    eigfaces, mean = calculate_eigenfaces(T_matrix)
    corr_mean = np.mean(T_matrix, axis=1).reshape(-1, 1)

    assert np.allclose(mean, corr_mean)

    num_pixels = T_matrix.shape[0]
    assert eigfaces.shape[0] == num_pixels

def test_recognise_input_face_finds_closest():
    training_weights = np.array([
        [1.0, 1.0], # Person 1 weight vector
        [5.0, 5.0], # Person 2 weight vector
        [9.0, 9.0], # Person 3 weight vector
    ])

    labels = ["Person1", "Person2", "Person3"]

    weight_vector = np.array([4.5, 4.5])
    smallest_distance = recognise_input_face(training_weights, weight_vector, labels)

    assert smallest_distance[1] == labels[1]
    assert np.allclose(smallest_distance[0], np.linalg.norm(weight_vector - training_weights[1]))

def test_load_input_face_path(tmp_path):
    test_img = Image.new('L', (2,2))
    img_path = tmp_path / "s1.pgm"
    test_img.save(img_path)

    image_vector, label = load_input_face(str(img_path))

    assert image_vector.shape == (4,)
    assert label == "1"

def test_load_input_face_fileobject():
    test_img = Image.new('L', (200, 200), color=100)
    buffer = io.BytesIO()
    test_img.save(buffer, format='PNG')
    buffer.seek(0)

    filestorage = FileStorage(
        stream=buffer,
        filename="upload.png",
        content_type="image/png",
    )

    image_vector, label = load_input_face(filestorage)

    assert image_vector.shape == (92 * 112,)   # flatten

    assert label == "upload.png"



    


