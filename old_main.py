import cv2
import numpy as np
import face_recognition

# Load reference images
img_arijit = face_recognition.load_image_file('img1.jpg')
img_rohit = face_recognition.load_image_file('rohit.jpg')
img_virat = face_recognition.load_image_file('virat.jpg')
img_ANSU= face_recognition.load_image_file('ANSU.jpg')

# Convert images to RGB
img_arijit = cv2.cvtColor(img_arijit, cv2.COLOR_BGR2RGB)
img_rohit = cv2.cvtColor(img_rohit, cv2.COLOR_BGR2RGB)
img_virat = cv2.cvtColor(img_virat, cv2.COLOR_BGR2RGB)
img_ANSU = cv2.cvtColor(img_ANSU, cv2.COLOR_BGR2RGB)

# Find face locations and encodings for reference images
face_location_arijit = face_recognition.face_locations(img_arijit)[0]
face_location_rohit = face_recognition.face_locations(img_rohit)[0]
face_location_virat = face_recognition.face_locations(img_virat)[0]
face_location_ANSU = face_recognition.face_locations(img_ANSU)[0]

encode_arijit = face_recognition.face_encodings(img_arijit)[0]
encode_rohit = face_recognition.face_encodings(img_rohit)[0]
encode_virat = face_recognition.face_encodings(img_virat)[0]
encode_ANSU = face_recognition.face_encodings(img_ANSU)[0]

# Open a video capture
video_capture = cv2.VideoCapture(0)

# Set frame dimensions
frame_width = 930
frame_height = 630
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Set desired FPS
desired_fps = 60
video_capture.set(cv2.CAP_PROP_FPS, desired_fps)

while True:
    # Start time for performance measurement
    start_time = cv2.getTickCount()

    # Read frames from the video
    ret, frame = video_capture.read()

    # Convert the frame to RGB for face_recognition
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    # Loop through each face in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the current face with the reference faces
        matches_arijit = face_recognition.compare_faces([encode_arijit], face_encoding)
        matches_rohit = face_recognition.compare_faces([encode_rohit], face_encoding)
        matches_virat = face_recognition.compare_faces([encode_virat], face_encoding)
        matches_ANSU = face_recognition.compare_faces([encode_ANSU], face_encoding)

        name = "Unknown"

        # If a match is found, display the corresponding name
        if True in matches_arijit:
            name = "Arijit"
        elif True in matches_rohit:
            name = "Rohit"
        elif True in matches_virat:
            name = "Virat"
        elif True in matches_ANSU:
            name = "ANSU"

        # Draw a rectangle and display the name on the frame
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 1)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0,0,255), 1)

    # Display the processed frame in a window named "Face Recognition"
    cv2.imshow("Face Recognition", frame)

    # Calculate processing time for this frame
    end_time = cv2.getTickCount()
    total_time = (end_time - start_time) / cv2.getTickFrequency()

    # Calculate time to wait before processing the next frame
    delay_time = max(1, int((1.0 / desired_fps - total_time) * 1000))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(delay_time) & 0xFF == ord("q"):
        break

# Release the video capture object
video_capture.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
