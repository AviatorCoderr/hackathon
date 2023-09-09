import streamlit as st
import cv2
import numpy as np
import os
import csv
from PIL import Image
from datetime import datetime

def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright


def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img


def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr


def main_loop():
    st.title("OpenCV Demo App")
    st.subheader("This app allows you to play with Image filters!")
    st.text("We use OpenCV and Streamlit for this demo")

    blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    processed_image = blur_image(original_image, blur_rate)
    processed_image = brighten_image(processed_image, brightness_amount)

    if apply_enhancement_filter:
        processed_image = enhance_details(processed_image)

    st.text("Original Image vs Processed Image")
    st.image([original_image, processed_image])


if __name__ == '__main__':
    main_loop()

def distance(v1, v2):
	return np.sqrt(((v1-v2)**2).sum())


def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]


#Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './data/'

face_data = []
labels = []

class_id = 0
names = {}

# Data Preparation

f = open('Names.csv', 'w+', newline='')
Inwriter = csv.writer(f)

for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):

		#Create a mapping btw class_id and name
		names[class_id] = fx[:-4]
		print("Loaded "+fx)

		# now = datetime.now()
		# current_date = now.strftime("%Y-%m-%d")
		# f = open(current_date+'.csv', 'w+', newline='')
		# Inwriter = csv.writer(f)
		# current_time = now.strftime("%H-%M-%S")
		# Inwriter.writerow([fx, current_time])



		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)

		#Create Labels for the class
		target = class_id*np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

my_set = {}

while True:
	ret, frame = cap.read()
	if ret == False:
		continue

	faces = model.detectMultiScale(frame,1.3, 5)
	if(len(faces) == 0):
		continue

	for face in faces:
		x, y, w, h = face

		#Get the face ROI
		offset = 10
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section = cv2.resize(face_section, (100, 100))

		#Predicted Label
		out = knn(trainset, face_section.flatten())

		#Display on the screen the name and rectangle around it
		pred_name = names[int(out)]
		my_set.add(pred_name)


		cv2.putText(frame, pred_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2, cv2.LINE_AA)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 2)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
current_time = now.strftime("%H-%M-%S")
Inwriter.writerow([my_set, current_time])

cap.release()
cv2.destroyAllWindows()
