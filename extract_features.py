import os
import numpy as np
import cv2
import dlib

predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# Load Dlib models
face_detector = dlib.get_frontal_face_detector() # type: ignore
shape_predictor = dlib.shape_predictor(predictor_path) # type: ignore
face_descriptor = dlib.face_recognition_model_v1(face_rec_model_path) # type: ignore

def extract_image_features_dlib(img_or_path):
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path)
    else:
        img = img_or_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray)

    if len(rects) == 0:
        raise ValueError("No face found")

    shape = shape_predictor(gray, rects[0])
    face_embedding = face_descriptor.compute_face_descriptor(img, shape)
    return np.array(face_embedding, dtype=np.float32)

def extract_multiple_face_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray)
    results = []
    for rect in rects:
        shape = shape_predictor(gray, rect)
        face_embedding = face_descriptor.compute_face_descriptor(img, shape)
        results.append((rect, np.array(face_embedding, dtype=np.float32)))
    return results

def load_features(data_path, mode='image'):
    from collections import Counter
    X, y = [], []
    for person in os.listdir(data_path):
        person_path = os.path.join(data_path, person)
        if not os.path.isdir(person_path):
            continue
        for file in os.listdir(person_path):
            file_path = os.path.join(person_path, file)
            try:
                features = extract_image_features_dlib(file_path)
                X.append(features)
                y.append(person)
            except Exception as e:
                print(f"❌ Skipped {file_path}: {e}")
    print("Class counts:", dict(Counter(y)))
    return np.array(X), np.array(y)
