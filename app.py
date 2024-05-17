from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import face_recognition
import os
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Dictionary to store known faces and their names
known_face_encodings = []
known_face_names = []

# Function to load images and add to the known faces
def load_and_encode_all_images():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            name = os.path.splitext(filename)[0]
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image = face_recognition.load_image_file(file_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)

# Preload known faces
load_and_encode_all_images()

def recognize_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            # Assign the name of the uploaded image to the detected face
            name = known_face_names[best_match_index]

        # Get the width and height of the text
        (text_width, text_height), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)  # Adjust font size here

        # Draw a purple rectangle behind the text, adjust padding based on text size
        cv2.rectangle(frame, (left, bottom), (left + text_width + 12, bottom + text_height + 31), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 1)
        
        # Change the font to FONT_HERSHEY_SIMPLEX
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Put the name on the frame with the new font and size
        cv2.putText(frame, name, (left + 6, bottom + text_height + 18), font, 1.2, (255, 255, 255), 2)  # Adjust position and thickness

    return frame



def video_stream():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        processed_frame = recognize_faces(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files or 'name' not in request.form:
        return redirect(request.url)

    file = request.files['image']
    name = request.form['name']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        filename = secure_filename(name) + '.' + file_extension
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Reload the image and encode it
        #load_and_encode_image(file_path, name)
        
        # Redirect to the index page
        return redirect(url_for('index'))
    else:
        return "Unsupported file type. Only PNG, JPG, and JPEG files are allowed."



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
