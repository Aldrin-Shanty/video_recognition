import os
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from audio_features import load_audio_features

def train_and_save_audio_model():
    data_path = "Audio_Data"
    X, y = load_audio_features(data_path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    os.makedirs("models", exist_ok=True)
    with open('models/audio_model.pkl', 'wb') as f:
        pickle.dump((scaler, clf), f)

    print(f"[✓] Trained and saved audio model for {len(set(y))} users.")

if __name__ == "__main__":
    train_and_save_audio_model()
