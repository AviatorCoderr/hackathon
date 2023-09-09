import face_recognition

# Load and encode known faces
known_face_encodings = []
known_face_names = []

# Load and encode known faces
person1_image = face_recognition.load_image_file("person1.jpg")
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]
known_face_encodings.append(person1_face_encoding)
known_face_names.append("Person 1")

person2_image = face_recognition.load_image_file("person2.jpg")
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]
known_face_encodings.append(person2_face_encoding)
known_face_names.append("Person 2")

# Load an unknown image
unknown_image = face_recognition.load_image_file("unknown.jpg")

# Find face locations in the unknown image
face_locations = face_recognition.face_locations(unknown_image)

# Encode the faces in the unknown image
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Perform face recognition
for face_encoding in face_encodings:
    # Compare the face encoding to known face encodings
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"  # Default name if no match is found

    # Use the first match (if any)
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    print(f"Found: {name}")
