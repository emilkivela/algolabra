from services.eigenfaces import load_dataset_faces, load_input_face, calculate_eigenfaces, get_input_weight, get_training_weights, recognise_input_face


T_matrix = load_dataset_faces('./images/data/')

eigenfaces, mean = calculate_eigenfaces(T_matrix)
print(eigenfaces.shape)
weight_vector, test_label = get_input_weight('./images/inputs/s1', mean, eigenfaces)
training_weights, labels = get_training_weights('./images/data/', eigenfaces, mean)
guess, distance = recognise_input_face(training_weights, weight_vector, test_label, labels)
print(guess, distance)
