from src.services.utils import load_dataset_faces
from PIL import Image
import numpy as np


def test_load_dataset_faces_returns_correct_shaped_matrix(tmp_path):
    testperson = tmp_path / "person1"
    testperson.mkdir()

    image1 = Image.fromarray(np.ones((5,5), dtype=np.uint8))
    image2 = Image.fromarray(np.full((5,5), 2, dtype=np.uint8))

    image1.save(testperson / "1.pgm")
    image2.save(testperson / "2.pgm")

    t_matrix = load_dataset_faces(tmp_path)

    assert t_matrix.shape == (25,2)
