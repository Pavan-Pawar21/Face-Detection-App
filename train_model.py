import cv2
import os
import numpy as np

dataset_path = "dataset"
labels = []
faces = []
label_map = {}
current_label = 0

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        faces.append(img)
        labels.append(current_label)

    current_label += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("face_model.yml")

print("Model trained successfully!")
print("Label map:", label_map)
