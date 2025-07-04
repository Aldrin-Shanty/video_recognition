import os
import numpy as np
import librosa

def extract_audio_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean.astype(np.float32)

def load_audio_features(data_path):
    from collections import Counter
    X, y = [], []
    for person in os.listdir(data_path):
        person_path = os.path.join(data_path, person)
        if not os.path.isdir(person_path):
            continue
        for file in os.listdir(person_path):
            if not file.endswith(".wav"):
                continue
            file_path = os.path.join(person_path, file)
            try:
                features = extract_audio_features(file_path)
                X.append(features)
                y.append(person)
            except Exception as e:
                print(f"❌ Skipped {file_path}: {e}")
    print("Class counts:", dict(Counter(y)))
    return np.array(X), np.array(y)
