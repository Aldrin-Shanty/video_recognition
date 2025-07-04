import streamlit as st
import cv2
import numpy as np
import pickle
import dlib
import tempfile
import time
import random
import sounddevice as sd
import soundfile as sf
from PIL import Image
from collections import Counter
from extract_features import extract_multiple_face_features
from audio_features import extract_audio_features

# ------------------- Load Models -------------------
with open("models/image_model.pkl", "rb") as f:
    scaler, clf = pickle.load(f)

with open("models/audio_model.pkl", "rb") as f:
    audio_scaler, audio_clf = pickle.load(f)

face_detector = dlib.get_frontal_face_detector() # type: ignore

# ------------------- Prediction Functions -------------------
def predict_faces_in_frame(frame):
    results = []
    try:
        detections = extract_multiple_face_features(frame)
        for rect, features in detections:
            scaled = scaler.transform([features])
            pred = clf.predict(scaled)[0]
            results.append((rect, pred))
    except Exception as e:
        st.warning(f"Image prediction error: {e}")
    return results

def predict_from_audio(file_path):
    try:
        features = extract_audio_features(file_path)
        scaled = audio_scaler.transform([features])
        pred = audio_clf.predict(scaled)[0]
        return pred
    except Exception as e:
        st.warning(f"Audio prediction error: {e}")
        return None

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Recognizer App", layout="centered")
st.title("🧠 Face & Voice Recognizer")

mode = st.sidebar.radio("Choose a mode:", ["📁 Detect from Saved Video", "📡 Real-Time Live Detection", "🎤 Record for 3 Seconds"])

# ------------------- Mode 1: Saved Video -------------------
if mode == "📁 Detect from Saved Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        st.video(uploaded_file)

        if st.button("🔍 Detect"):
            if not frames:
                st.error("⚠️ No frames found in video.")
            else:
                sampled = random.sample(frames, min(10, len(frames)))
                all_preds = []
                for f in sampled:
                    res = predict_faces_in_frame(f)
                    for _, p in res:
                        all_preds.append(p)

                if all_preds:
                    counter = Counter(all_preds)
                    result = counter.most_common(1)[0][0]
                    st.success(f"🧠 Detected in Video: {result}")
                else:
                    st.warning("😶 No face detected.")

# ------------------- Mode 2: Live Detection -------------------
elif mode == "📡 Real-Time Live Detection":
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    cap = None
    prev_prediction = None

    if run:
        cap = cv2.VideoCapture(0)

    while run and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        detections = extract_multiple_face_features(small_frame)
        current_preds = []

        for rect, features in detections:
            scaled = scaler.transform([features])
            pred = clf.predict(scaled)[0]
            current_preds.append(pred)

            x, y, w, h = rect.left()*2, rect.top()*2, rect.width()*2, rect.height()*2
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, pred, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(rgb)

        if current_preds:
            counter = Counter(current_preds)
            prediction = ", ".join(name for name, _ in counter.most_common())
        else:
            prediction = "No face detected"

        if prediction != prev_prediction:
            st.write(f"🔍 Detected: {prediction}")
            prev_prediction = prediction

    if cap:
        cap.release()

# ------------------- Mode 3: Record for 3 Seconds -------------------
elif mode == "🎤 Record for 3 Seconds":
    if st.button("🎬 Start Recording"):
        st.info("Recording starts in 3 seconds...")
        time.sleep(3)

        st.warning("Recording now for 3 seconds...")
        cap = cv2.VideoCapture(0)
        frames = []
        start = time.time()

        fs = 44100
        audio = sd.rec(int(3 * fs), samplerate=fs, channels=1)

        frame_placeholder = st.empty()

        while time.time() - start < 3:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(display, caption="Recording...", use_container_width=True)


        cap.release()
        sd.wait()
        sf.write("recorded_audio.wav", audio, fs)

        st.info("Processing...")

        # Image Prediction
        image_pred = "No face"
        if frames:
            sampled = random.sample(frames, min(10, len(frames)))
            all_preds = []
            for f in sampled:
                res = predict_faces_in_frame(f)
                for _, p in res:
                    all_preds.append(p)

            if all_preds:
                counter = Counter(all_preds)
                image_pred = counter.most_common(1)[0][0]

        # Audio Prediction
        audio_pred = predict_from_audio("recorded_audio.wav")

        st.success(f"🧠 Image Prediction: {image_pred}")
        st.success(f"🎤 Audio Prediction: {audio_pred if audio_pred else 'No audio prediction'}")
