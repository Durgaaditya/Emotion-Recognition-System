# 🎙️ Speech Emotion Recognition (SER) System

This project is a Python-based Speech Emotion Recognition (SER) system that detects human emotions like **happy**, **sad**, **angry**, **neutral**, and more from audio recordings using deep learning and audio signal processing techniques.

---

##  Features

- Emotion detection from `.wav` audio files
- Trained on the **RAVDESS** dataset
- Feature extraction using **MFCCs** (Mel-frequency cepstral coefficients)
- Deep Neural Network built with **TensorFlow/Keras**
- Emotion distribution visualization with **Seaborn**
- Fully runnable from **VS Code only**

---

##  Project Structure

emotion_recognition/
 data/  .wav files
models/ 
test_audio.wav 
emotion_recognition.py 
requirements.txt 
README.md # This file


---

##  Dataset

We use the [**RAVDESS Speech Dataset**](https://zenodo.org/record/1188976).

 Download:
- [Audio_Speech_Actors_01-24.zip](https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip)

 Extract and move all `.wav` files into the `data/` folder.

---

##  How It Works

1. Loads RAVDESS audio files and extracts MFCC features.
2. Visualizes emotion distribution in a bar chart.
3. Trains a deep neural network for emotion classification.
4. Saves the trained model in the `models/` folder.
5. Predicts the emotion of any new `.wav` file (e.g., `test_audio.wav`).

---

##  How to Run (VS Code Only)

1. **Clone or Download** the project
2. **Install Dependencies**:

```bash
pip install -r requirements.txt

