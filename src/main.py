import numpy as np
import os
from PIL import Image

def load_faces(dataset_path):
    data_matrix = []
    labels = []
    for person_dir in os.listdir(dataset_path):
        for file in os.listdir(os.path.join(dataset_path, person_dir)):
            #print(file)
            img = Image.open(os.path.join(dataset_path, person_dir,file))
            img_vector = np.array(img).flatten()
            data_matrix.append(img_vector)
            labels.append(f"Person {person_dir[1:]}")

    T_matrix = np.array(data_matrix).T
    labels = np.array(labels)

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
    
    #eigface1 = eigfaces[0]
    #print(eigface1)
    #print(len(eigface1))
    #build_eigface(eigface1, 1)

    for i in range(len(eigfaces)):
        build_eigface(eigfaces[i], i)
        


    

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


def hotelling_deflation(A, eigvalue, eigvector):
    eigvector = eigvector / np.linalg.norm(eigvector)
    next_A = (A-(eigvalue*np.outer(eigvector, eigvector)))
    return next_A

def rayleigh_quotient(vector, M):
    eig = (np.dot(vector.T, np.dot(M, vector))) / (np.dot(vector.T, vector))
    return eig


def power_iteration(A, n):
    b_k = np.random.rand(A.shape[0])

    for _ in range(n):

        b_k1 = np.dot(A, b_k)
        b_k_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k_norm
    
    return b_k

    
def get_mean_face(mean_vector):
    mean_image = mean_vector.reshape((112,92))
    mean_image_norm = (mean_image - mean_image.min()) / (mean_image.max() - mean_image.min())
    mean_image_uint8 = (mean_image_norm * 255).astype(np.uint8)

    face_img = Image.fromarray(mean_image_uint8, mode='L')
    face_img.show()
    face_img.save("./images/mean_face_image.png")

def build_eigface(eigvector, name):
    eigvector = eigvector.reshape((112, 92))
    eigvector = eigvector - np.min(eigvector)
    eigvector = eigvector / np.max(eigvector)
    img = (eigvector * 255).astype(np.uint8)

    eigenface = Image.fromarray(img, mode='L')
    eigenface.save(f"./images/eigenfaces/eigenface{name}.png")




load_faces('./images/data/')


