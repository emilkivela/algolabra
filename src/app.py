import os
import time
import numpy as np
from flask import Flask, render_template, request, flash, redirect, url_for
from src.services.utils import convert_pmg, load_dataset_faces, calculate_treshold
from src.services.eigenfaces import (calculate_eigenfaces, get_input_weight, 
                                     get_training_weights, recognise_input_face)

app = Flask(__name__)

start = time.time()
T_matrix = load_dataset_faces('./static/images/data/')
faceloadtime = time.time()
eigenfaces, mean = calculate_eigenfaces(T_matrix)
eigtime = time.time()
training_weights, labels = get_training_weights('./static/images/data/', eigenfaces, mean)
treshold = calculate_treshold(training_weights, labels, eigenfaces, mean)
print(np.shape(training_weights), labels)
weighttime = time.time()

input_files = []
for file in os.listdir('./static/images/inputs/'):
    base64 = convert_pmg('./static/images/inputs/'+file)
    input_files.append(base64)
end = time.time()

print("Load dataset faces time:", (faceloadtime-start))
print("eigface calc time:", (eigtime-start))
print("Weight time:", (weighttime-start))
print("Whole preparation time: ", (end-start))

@app.route("/")
def index():
    """
    Metodi, joka palauttaa ohjelman etusivun.
    """
    #error = None
    return render_template("index.html", face_files = input_files)

@app.route("/recognise", methods=["POST"])
def recognise_face():
    """
    Metodi, joka näyttää syötekuvan ja keneksi ohjelma sen tunnisti. Näyttää myös kaikki harjoitusdatan kuvat arvatusta henkilöstä.
    """
    start = time.time()
    error = None
    if "upload" in request.files:
        input_face_path = request.files["upload"]
        if input_face_path.content_type[0:5] != "image":
            error = "Syötä vain kuvatiedostoja"
            return render_template("index.html", face_files = input_files, error = error)
    elif "faces" in request.form:
        file = request.form["faces"]
        input_face_path = "./static/images/inputs/"+file

    input_image = convert_pmg(input_face_path)
    input_image = input_image["base64"]

    weight_vector, true_label = get_input_weight(input_face_path, mean, eigenfaces)
    guess = recognise_input_face(training_weights, weight_vector, labels)

    if guess[0] > treshold:
        return render_template("not_recognised.html", input_image=input_image, true_label=true_label)

    guess_pictures = []
    path = './static/images/data/'+guess[1]+'/'
    for img in os.listdir(path):
        converted = convert_pmg(path+img)
        guess_pictures.append(converted["base64"])

    end = time.time()
    print("Recognition time: ", (end-start))
    
    return render_template("guess.html", guess=guess[1][1:], distance=guess[0], 
                                         input_image=input_image, true_label=true_label, guesspics= guess_pictures)

@app.route("/mean")
def mean_face():
    """
    Metodi, joka palauttaa sivun jossa näytetään kuvamatriisin keskiarvovektorista rakennettu kuva.
    """
    return render_template("mean.html")

@app.route("/eigenfaces")
def eigenface():
    """
    Metodi, joka palauttaa sivun jossa näytetään kaikki lasketut ominaiskasvot kuvina.
    """
    eigenfaces = []
    for eigface in os.listdir('./static/images/eigenfaces/'):
        eigenfaces.append('/static/images/eigenfaces/'+eigface)
    eigenfaces = sorted(eigenfaces)
    return render_template("eigenfaces.html", eigenfaces=eigenfaces)