import os
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from extract_features import load_features

def train_and_save_image_model():
    data_path = "Image_Data"
    X, y = load_features(data_path, mode='image')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    os.makedirs("models", exist_ok=True)
    with open('models/image_model.pkl', 'wb') as f:
        pickle.dump((scaler, clf), f)

    print(f"[✓] Trained and saved image model for {len(set(y))} users.")

if __name__ == "__main__":
    train_and_save_image_model()
