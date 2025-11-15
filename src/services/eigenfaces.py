import numpy as np
import os
from PIL import Image
from .formulas import hotelling_deflation, rayleigh_quotient, power_iteration 

def load_dataset_faces(dataset_path):
    data_matrix = []
    for person_dir in os.listdir(dataset_path):
        for file in os.listdir(os.path.join(dataset_path, person_dir)):
            #print(file)
            img = Image.open(os.path.join(dataset_path, person_dir,file))
            img_vector = np.array(img, dtype=np.float32).flatten()
            data_matrix.append(img_vector)

    T_matrix = np.array(data_matrix).T

    return T_matrix

def calculate_eigenfaces(T_matrix):
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
    test_img, test_label = load_input_face(dataset_path)
    centered_img = test_img-mean.flatten()

    weight_vector = []

    weight_vector = np.dot(eigfaces.T, centered_img)
    return weight_vector, test_label

def get_training_weights(dataset_path, eigfaces, mean):
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
        labels.append(f"Person {person_dir[1:]}")
    
    return training_weights, labels

def recognise_input_face(training_weights, weight_vector, test_label, labels):

    smallest_distance = (float("inf"), "class label")
    for i in range(len(training_weights)):
        distance = np.linalg.norm(weight_vector-training_weights[i])
        if distance < smallest_distance[0]:
            smallest_distance = distance,labels[i]
    
    test = f"Guess: {smallest_distance[1]}, True: {test_label}"
    return test, smallest_distance[0]

def load_input_face(img_path):
    img = Image.open(img_path)
    img_vector = np.array(img, np.float32).flatten()
    label = f"Person {os.path.basename(img_path)[1:]}"
    return img_vector, label



    

def get_eigs(C, k):
    eigs = []
    vectors = []
    for _ in range(k):
        eigvector = power_iteration(C, 50)
        eigvalue = rayleigh_quotient(eigvector, C)
        eigs.append(eigvalue)
        vectors.append(eigvector)
        C = hotelling_deflation(C, eigvalue, eigvector)
    return eigs, vectors

def get_mean_face(mean_vector):
    mean_image = mean_vector.reshape((112,92))
    mean_image_norm = (mean_image - mean_image.min()) / (mean_image.max() - mean_image.min())
    mean_image_uint8 = (mean_image_norm * 255).astype(np.uint8)

    face_img = Image.fromarray(mean_image_uint8, mode='L')
    face_img.show()
    face_img.save("../images/mean_face_image.png")

def build_eigface(eigvector, name):
    eigvector = eigvector.reshape((112, 92))
    eigvector = eigvector - np.min(eigvector)
    eigvector = eigvector / np.max(eigvector)
    img = (eigvector * 255).astype(np.uint8)

    eigenface = Image.fromarray(img, mode='L')
    eigenface.save(f"./images/eigenfaces/eigenface{name}.png")
