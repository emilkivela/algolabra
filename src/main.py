import numpy as np
import os
from PIL import Image

def load_faces(dataset_path):
    for person_dir in os.listdir(dataset_path):
        for file in os.listdir(os.path.join(dataset_path, person_dir)):
            print(file)
            img = Image.open(file)
            

load_faces('./data/')


