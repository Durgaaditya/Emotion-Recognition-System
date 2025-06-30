import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

# Emotion dictionary based on RAVDESS filename
emotion_dict = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print("Error processing:", file_path, "Error:", e)
        return None

def load_data():
    features, labels = [], []
    for file in os.listdir("data"):
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            emotion = emotion_dict.get(emotion_code)
            feature = extract_features(os.path.join("data", file))
            if feature is not None:
                features.append(feature)
                labels.append(emotion)
    return np.array(features), np.array(labels)

def build_model(input_shape, output_shape):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(output_shape, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model():
    X, y = load_data()

    # 🧾 Step 8: Visualize emotion label distribution
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    print("\n📊 Showing emotion distribution...")
    # Convert labels into a DataFrame
    label_df = pd.DataFrame(y, columns=["Emotion"])

    # Set up the size of the chart
    plt.figure(figsize=(10, 5))

    # Create a count plot
    sns.countplot(data=label_df, x="Emotion", palette="magma")

    # Add title and rotate x-axis labels
    plt.title("Emotion Distribution in Dataset", fontsize=14)
    plt.xticks(rotation=45)

    # Fit layout and display
    plt.tight_layout()
    plt.show()

    # 🧠 Continue training as before
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split

    le = LabelEncoder()
    y_encoded = to_categorical(le.fit_transform(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = build_model(X.shape[1], y_encoded.shape[1])
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    model.save("models/emotion_model.h5")
    print("✅ Model training complete.")
    return le

def predict_emotion(file_path, encoder):
    model = load_model("models/emotion_model.h5")
    feature = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(feature)
    return encoder.inverse_transform([np.argmax(prediction)])[0]

if __name__ == "__main__":
    print("Training model...")
    encoder = train_model()
    print("Training complete.")

    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        print(f"Testing on {test_file}...")
        result = predict_emotion(test_file, encoder)
        print("Predicted Emotion:", result)
    else:
        print("No test_audio.wav found for testing.")
