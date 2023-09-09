import cv2
import numpy as np
import os
import csv
from datetime import datetime
import streamlit as st

def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from the test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find the max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]

# Streamlit UI
st.title("Face Recognition App")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    # Initialize camera and face detection model
    cap = cv2.VideoCapture(0)
    model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    # Load saved face data
    dataset_path = './data/'
    face_data = []
    labels = []
    class_id = 0
    names = {}

    for fx in os.listdir(dataset_path):
        if fx.endswith('.npy'):
            names[class_id] = fx[:-4]
            print("Loaded " + fx)

            data_item = np.load(dataset_path + fx)
            face_data.append(data_item)

            target = class_id * np.ones((data_item.shape[0],))
            class_id += 1
            labels.append(target)

    face_dataset = np.concatenate(face_data, axis=0)
    face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

    trainset = np.concatenate((face_dataset, face_labels), axis=1)

    while True:
        ret, frame = cap.read()
        if ret == False:
            continue

        faces = model.detectMultiScale(frame, 1.3, 5)
        if len(faces) == 0:
            continue

        for face in faces:
            x, y, w, h = face

            # Get the face ROI
            offset = 10
            face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
            face_section = cv2.resize(face_section, (100, 100))

            # Predicted Label
            out = knn(trainset, face_section.flatten())

            # Display on the screen the name and rectangle around it
            pred_name = names[int(out)]

            cv2.putText(frame, pred_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 2)

        # Display the processed frame in Streamlit
        st.image(frame, channels="BGR", use_column_width=True)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
