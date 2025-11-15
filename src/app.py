import os
from flask import Flask, render_template, request
from src.services.eigenfaces import (
    load_dataset_faces, load_input_face, calculate_eigenfaces, 
    get_input_weight, get_training_weights, recognise_input_face
)

app = Flask(__name__)

face_files = os.listdir('./images/inputs')



#eigenfaces, mean = calculate_eigenfaces(T_matrix)
#print(eigenfaces.shape)
#weight_vector, test_label = get_input_weight('./images/inputs/s2', mean, eigenfaces)
#training_weights, labels = get_training_weights('./images/data/', eigenfaces, mean)
#guess, distance = recognise_input_face(training_weights, weight_vector, test_label, labels)
#print(guess, "Distance: ", distance)

@app.route("/")
def index():
    return render_template("index.html", face_files = face_files)

@app.route("/recognise", methods=["POST"])
def recognise_face():
    file = request.form["faces"]
    input_face_path = "./images/inputs/"+file

    T_matrix = load_dataset_faces('./images/data/')
    eigenfaces, mean = calculate_eigenfaces(T_matrix)
    weight_vector, test_label = get_input_weight(input_face_path, mean, eigenfaces)
    training_weights, labels = get_training_weights('./images/data/', eigenfaces, mean)
    guess, distance = recognise_input_face(training_weights, weight_vector, test_label, labels)
    print(guess, "Distance: ", distance)
    return guess