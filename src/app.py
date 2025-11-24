import os
from flask import Flask, render_template, request
from src.services.helper import convert_pmg
from src.services.eigenfaces import (
    load_dataset_faces, calculate_eigenfaces, 
    get_input_weight, get_training_weights, recognise_input_face
)

app = Flask(__name__)

T_matrix = load_dataset_faces('./static/images/data/')
eigenfaces, mean = calculate_eigenfaces(T_matrix)
training_weights, labels = get_training_weights('./static/images/data/', eigenfaces, mean)


input_files = []
for file in os.listdir('./static/images/inputs/'):
    base64 = convert_pmg('./static/images/inputs/'+file)
    input_files.append(base64)

@app.route("/")
def index():
    """
    Metodi, joka palauttaa ohjelman etusivun.
    """
    return render_template("index.html", face_files = input_files)

@app.route("/recognise", methods=["POST"])
def recognise_face():
    """
    Metodi, joka näyttää syötekuvan ja keneksi ohjelma sen tunnisti. Näyttää myös kaikki harjoitusdatan kuvat arvatusta henkilöstä.
    """
    if "upload" in request.files:
        input_face_path = request.files["upload"]
    elif "faces" in request.form:
        file = request.form["faces"]
        input_face_path = "./static/images/inputs/"+file
    
    input_image = convert_pmg(input_face_path)
    input_image = input_image["base64"]

    weight_vector, true_label = get_input_weight(input_face_path, mean, eigenfaces)
    guess = recognise_input_face(training_weights, weight_vector, labels)
    
    guess_pictures = []
    path = './static/images/data/'+guess[1]+'/'
    for img in os.listdir(path):
        converted = convert_pmg(path+img)
        guess_pictures.append(converted["base64"])
    
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