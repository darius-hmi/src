import face_recognition
import cv2

person1_image = face_recognition.load_image_file("Shahin.jpg")
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

person2_image = face_recognition.load_image_file("Saboo.jpg")
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]

person3_image = face_recognition.load_image_file("Soroosh.jpg")
person3_face_encoding = face_recognition.face_encodings(person3_image)[0]

person4_image = face_recognition.load_image_file("Adan.jpg")
person4_face_encoding = face_recognition.face_encodings(person4_image)[0]

known_face_encodings = [
    person1_face_encoding,
    person2_face_encoding,
    person3_face_encoding,
    person4_face_encoding
]

# Store the names of the people in a list
known_face_names = [
    "Shahin",
    "Saboo",
    "Soroosh",
    "Adan"
]

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_locations = face_recognition.face_locations(frame, model="hog")
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding with all known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # Set a default label as "Unknown"
        label = "Unknown"

        # If a match is found, get the name of the matched person
        if True in matches:
            match_index = matches.index(True)
            label = known_face_names[match_index]

        # Draw a rectangle around the face and label it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with face recognition
    cv2.imshow('Face Recognition', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()