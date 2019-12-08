import face_recognition
import cv2

# Get the WebCam reference
video_capture = cv2.VideoCapture(0)

# Initialize variables
face_locations = []

while True:
    # Get a strem frame
    ret, frame = video_capture.read()

    # Resize the stream frame
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Search for face and encoding in the stream frame
    face_locations = face_recognition.face_locations(small_frame, model="cnn")

    # Display the results
    for top, right, bottom, left in face_locations:
        # Scale the face regions to correspond with stream resizing
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Store the image face region
        face_image = frame[top:bottom, left:right]

        # Blur the face
        face_image = cv2.GaussianBlur(face_image, (99, 99), 30)

        # Put the blurred face region back into the frame image
        frame[top:bottom, left:right] = face_image

    # Display the image
    cv2.imshow('Blur Faces:', frame)

    # 'q' to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the handle
video_capture.release()
cv2.destroyAllWindows()
