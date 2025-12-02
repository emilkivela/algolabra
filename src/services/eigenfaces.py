import numpy as np
import os
from PIL import Image
from .formulas import hotelling_deflation, rayleigh_quotient, power_iteration 
#from .utils import load_input_face


def calculate_eigenfaces(T_matrix):
    """
    Funktio joka keskittää datan, laskee ominaisarvot ja -vektorit sekä ominaiskasvot.
    """
    mean = np.mean(T_matrix, axis=1)
    mean = mean[:, np.newaxis]
    A_matrix = T_matrix - mean

    covariance_matrix = np.dot(A_matrix.T, A_matrix)
    matrix_rank = np.linalg.matrix_rank(covariance_matrix)
    eigs, eigvectors = get_eigs(covariance_matrix, matrix_rank)
    variance_threshold = 0.95
    k = 1
    while ((sum(eigs[:k])) / (sum(eigs))) <= variance_threshold:
        k += 1

    eigfaces = []

    for i in range(k):
        eigfaces.append(np.dot(A_matrix, eigvectors[i]))

    eigfaces = np.array(eigfaces).T

    return eigfaces, mean

def get_input_weight(dataset_path, mean, eigfaces):
    """
    Laskee painovektorin kuvalle, joka on tarkoitus tunnistaa.
    """
    test_img, test_label = load_input_face(dataset_path)
    centered_img = test_img - mean.flatten()

    weight_vector = []

    weight_vector = np.dot(eigfaces.T, centered_img)
    return weight_vector, test_label

def get_training_weights(dataset_path, eigfaces, mean):
    """
    Laskee painovektorit kaikille harjoitussetin kuvien henkilöille.
    """
    labels = []
    training_weights = []
    for person_dir in os.listdir(dataset_path):
        person_weights = []
        for file in os.listdir(os.path.join(dataset_path, person_dir)):
            img = Image.open(os.path.join(dataset_path, person_dir,file))
            img_vector = np.array(img, np.float32).flatten()
            img_vector -= mean.flatten()
            w = np.dot(eigfaces.T, img_vector)
            person_weights.append(w)
        avg_weight = np.mean(person_weights, axis=0)
        training_weights.append(avg_weight)
        labels.append(person_dir)
    
    return training_weights, labels

def recognise_input_face(training_weights, weight_vector, labels):
    """
    Laskee pienimmän Euklidisen etäisyyden syötekuvan ja harjoituskuvien painovektorien välillä, ja luokittelee kasvot sen mukaan.
    """
    smallest_distance = (float("inf"), "class label")
    for i in range(len(training_weights)):
        distance = np.linalg.norm(weight_vector-training_weights[i])
        if distance < smallest_distance[0]:
            smallest_distance = distance,labels[i]
    return smallest_distance


def get_eigs(C, k):
    """
    Laskee matriisille C k-määrän suurimpia ominaisarvoja ja niitä vastaavia vektoreita.
    """
    eigs = []
    vectors = []
    for _ in range(k):
        eigvector = power_iteration(C, 50)
        eigvalue = rayleigh_quotient(eigvector, C)
        eigs.append(eigvalue)
        vectors.append(eigvector)
        C = hotelling_deflation(C, eigvalue, eigvector)
    return eigs, vectors


def load_input_face(img_path):
    """
    Muuntaa syötekuvan kuvavektoriksi ja antaa sille nimen.
    """
    if isinstance(img_path, str):
        img = Image.open(img_path)
        img_vector = np.array(img, np.float32).flatten()
        label = f"{os.path.basename(img_path)[1:-4]}"
    else:
        img = Image.open(img_path.stream).convert('L')
        img = img.resize((92, 112))
        img_vector = np.array(img, np.float32).flatten()
        label = f"{img_path.filename}"

    return img_vector, label

