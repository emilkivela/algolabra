import numpy as np

def hotelling_deflation(A, eigvalue, eigvector):
    eigvector = eigvector / np.linalg.norm(eigvector)
    next_A = (A-(eigvalue*np.outer(eigvector, eigvector)))
    
    return next_A

def rayleigh_quotient(vector, M):
    eig = (np.dot(vector.T, np.dot(M, vector))) / (np.dot(vector.T, vector))
    
    return eig

def power_iteration(A, n):
    vector = np.random.rand(A.shape[0])
    for _ in range(n):
        vector = np.dot(A, vector) / np.linalg.norm(np.dot(A, vector))
    
    return vector