import os
from flask import Flask, render_template, request
from src.services.helper import convert_pmg
from src.services.eigenfaces import (
    load_dataset_faces, load_input_face, calculate_eigenfaces, 
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
    return render_template("index.html", face_files = input_files)

@app.route("/recognise", methods=["POST"])
def recognise_face():
    file = request.form["faces"]
    input_face_path = "./static/images/inputs/"+file
    weight_vector, test_label = get_input_weight(input_face_path, mean, eigenfaces)
    guess = recognise_input_face(training_weights, weight_vector, test_label, labels)
    guess_pictures = []
    path = './static/images/data/'+guess[1]+'/'
    for img in os.listdir(path):
        converted = convert_pmg(path+img)
        guess_pictures.append(converted["base64"])
    input_image = convert_pmg(input_face_path)
    return render_template("guess.html", guess=guess[1], distance=guess[0], 
                                         input_image=input_image["base64"], true_label=file, guesspics= guess_pictures)

@app.route("/mean")
def mean_face():
    return render_template("mean.html")

@app.route("/eigenfaces")
def eigenface():
    eigenfaces = []
    for eigface in os.listdir('./static/images/eigenfaces/'):
        eigenfaces.append('/static/images/eigenfaces/'+eigface)
    eigenfaces = sorted(eigenfaces)
    return render_template("eigenfaces.html", eigenfaces=eigenfaces)