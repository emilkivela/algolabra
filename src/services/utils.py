import io
import base64
import os
import numpy as np
from PIL import Image
from .eigenfaces import get_input_weight

def load_dataset_faces(dataset_path):
    """
    Muuntaa harjoitussetin kuvat vektoreiksi ja tallettaa ne matriisiin, niin että jokainen sarake on yksi kuvavektori.
    """
    data_matrix = []
    for person_dir in os.listdir(dataset_path):
        for file in os.listdir(os.path.join(dataset_path, person_dir)):
            img = Image.open(os.path.join(dataset_path, person_dir,file))
            img_vector = np.array(img, dtype=np.float32).flatten()
            data_matrix.append(img_vector)

    T_matrix = np.array(data_matrix).T

    return T_matrix



def convert_pmg(image_path):
    """
    Muuntaa syötetyn kuvan base64-muotoon, jotta se voidaan esittää verkkosivulla riippumatta siitä,
    oliko alkuperäinen kuva .pmg-tiedosto vai ei. Tarkistaa alussa onko syöte polku tiedostoon vai FileObject.
    """
    if isinstance(image_path, str):
        img = Image.open(image_path)
        file_name = os.path.basename(image_path)
    else:
        img = Image.open(image_path.stream).convert('L')
        img = img.resize((92, 112))
        file_name = image_path.filename

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    file = {"filename" : file_name, "base64" : encoded}
    return file

def get_mean_face(mean_vector):
    """
    Luo kuvamatriisin keskiarvovektorista kuvan harjoitussetin kasvojen keskiarvoisesta kasvosta.
    """
    mean_image = mean_vector.reshape((112,92))
    mean_image_norm = (mean_image - mean_image.min()) / (mean_image.max() - mean_image.min())
    mean_image_uint8 = (mean_image_norm * 255).astype(np.uint8)

    face_img = Image.fromarray(mean_image_uint8, mode='L')
    face_img.show()
    face_img.save("../static/images/mean_face_image.png")

def build_eigface(eigvector, name):
    """
    Muuntaa lasketut ominaisvektorit kuviksi, eli ominaiskasvoiksi.
    """
    eigvector = eigvector.reshape((112, 92))
    eigvector = eigvector - np.min(eigvector)
    eigvector = eigvector / np.max(eigvector)
    img = (eigvector * 255).astype(np.uint8)

    eigenface = Image.fromarray(img, mode='L')
    eigenface.save(f"./static/images/eigenfaces/eigenface{name}.png")

def calculate_treshold(training_weigths, labels, eigenfaces, mean):
    """
    Laskee kynnysarvon jota käytetään määrittämään onko syötetty kuva nähty ennen vai ei. Kynnysarvoksi on valittu tässä tapauksessa
     etäisyyden keskiarvo johon on lisätty keskihajonta kerrottuna kahdella, kun etäisyys lasketaan harjoitussetissä olevan kuvan
     painovektorin ja saman henkilöluokan painovektorin välillä.
    """
    close_distances = []
    for person_dir in os.listdir('./static/images/data/'):
        idx = labels.index(person_dir)
        person_weight = training_weigths[idx]
        for file in os.listdir(os.path.join('./static/images/data/', person_dir)):
            input_weight = get_input_weight(os.path.join('./static/images/data/', person_dir, file), mean, eigenfaces)[0]
            dist = np.linalg.norm(input_weight-person_weight)
            close_distances.append(dist)
    weight_mean = np.mean(close_distances)
    weigth_std = np.std(close_distances)
    treshold = weight_mean + 2*weigth_std
    return treshold
