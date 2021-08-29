import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while True:
    succesful_frame_read, frame = webcam.read()

    # Convert image to greyscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw boxes around face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 4)

    cv2.imshow('Eddies face detector app!', frame)
    key = cv2.waitKey(1)

    # Stop if Q is pressed
    if key==81 or key==113:
        break

# Choose an image to detect faces in
# img = cv2.imread('Chris_hemsworth.jpg')
# img = cv2.imread('chan_family_group_picture.JPG')

"""
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

print(face_coordinates)

for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256),randrange(256), randrange(256)), 4)

cv2.imshow('Eddies face detector app!', img)
cv2.waitKey()
"""

print("Code Completed!")