from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import joblib
import time
import subprocess
import shutil

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# ================= INIT =================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs("uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

# ================= LOAD MODELS =================
print("Loading models...")

vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
pca = joblib.load("model/pca_model.pkl")
svm = joblib.load("model/svm_model.pkl")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

print("Models loaded!")

# ================= FEATURE EXTRACTION =================
def extract_features(frame):
    frame = cv2.resize(frame, (224,224))
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    return vgg.predict(frame, verbose=0).flatten()

# ================= PROCESS VIDEO =================
def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return "ERROR OPEN VIDEO", None, 0, 0, 0

    frames = []
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame is None:
            continue

        frames.append(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                continue

            try:
                feat = extract_features(face)
                feat = pca.transform([feat])
                pred = svm.predict(feat)[0]
                predictions.append(pred)
            except:
                pass

    cap.release()

    if len(frames) == 0:
        return "NO VIDEO FRAMES", None, 0, 0, 0

    if len(predictions) == 0:
        return "NO FACE DETECTED", None, 0, 0, 0

    # ===== RESULT =====
    morph = predictions.count(1)
    original = predictions.count(0)

    if morph > original:
        result = "MORPHED VIDEO"
        color = (0,0,255)
    else:
        result = "ORIGINAL VIDEO"
        color = (0,255,0)

    # ===== CONFIDENCE =====
    total = morph + original
    confidence = round((max(morph, original) / total) * 100, 2)

    # ===== SAVE TEMP FRAMES =====
    temp_folder = "temp_frames"
    os.makedirs(temp_folder, exist_ok=True)

    height, width, _ = frames[0].shape

    for i, frame in enumerate(frames):
        frame = cv2.resize(frame, (width, height))

        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (width,height), color, -1)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

        cv2.putText(frame, result, (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.imwrite(f"{temp_folder}/frame_{i:04d}.jpg", frame)

    # ===== FFMPEG ENCODING (FINAL FIX) =====
    filename = f"result_{int(time.time())}.mp4"
    output_path = os.path.join("static/results", filename)

    command = [
        r"C:\Users\yuvat\Desktop\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe",  # 👈 YOUR PATH
        "-y",
        "-framerate", "20",
        "-i", f"{temp_folder}/frame_%04d.jpg",
        "-r", "20",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-level", "3.0",
        "-movflags", "+faststart",
        output_path
    ]

    subprocess.run(command, check=True)

    # ===== CLEAN TEMP =====
    shutil.rmtree(temp_folder)

    print("Saved video:", output_path)
    print("File size:", os.path.getsize(output_path))

    return result, filename, morph, original, confidence

# ================= ROUTES =================
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['video']

    if file.filename == "":
        return "No file uploaded"

    path = os.path.join("uploads", file.filename)
    file.save(path)

    result, video_name, morph, original, confidence = predict_video(path)

    videos = sorted(os.listdir("static/results"), reverse=True)

    return render_template("dashboard.html",
                           result=result,
                           current_video=video_name,
                           videos=videos,
                           morph=morph,
                           original=original,
                           confidence=confidence)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)