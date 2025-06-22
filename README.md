# wav-classifier-mfcc-fft

This project implements two MATLAB scripts for classifying `.wav` audio files using digital signal processing and feature extraction techniques. It covers two parts:

---

## ✅ Part 1: Audio Classification Using MFCC/FFT
## 🎧 `classify.m` – MFCC/FFT-Based Audio Classifier

This script performs classification of audio files from three folders: `class_1`, `class_2`, and `unknown`.

### 🔍 What it does:
1. **Reads audio files** from the training folders (`class_1`, `class_2`) and unknown files.
2. **Extracts features** from each file using:
   - **MFCC** (Mel Frequency Cepstral Coefficients) or  
   - **FFT-based spectral features** (depending on your implementation).
3. **Computes similarity** between feature vectors using a selected distance metric:
   - Euclidean Distance
   - Cosine Similarity
   - Manhattan Distance
4. **Classifies unknown audio files** by comparing them with known class samples and assigning the closest match.
5. **Prints classification results**, showing the predicted class for each file.

### 📈 Use Case:
- General-purpose sound classification based on statistical signal features.
- Designed to work without needing pre-labeled test data.

---

## ✅ Part 2: Filter-Based Audio Classification
## 🚨 `filterDesign.m` – Filter-Based Emergency Sound Classifier

This script classifies emergency vehicle sounds (ambulance vs. firetruck) using bandpass filtering and energy ratio features.

### 🔍 What it does:
1. **Reads training data** from ambulance and firetruck folders.
2. **Analyzes frequency content** and applies three pre-defined **bandpass filters** designed to isolate different frequency bands.
3. **Extracts features** by computing:
   - **Energy in each filtered signal**, and
   - **Energy ratios** between bands (e.g., Filter 1 energy / Filter 2 energy).
4. **Trains a simple k-NN classifier** (`k=3`) using these energy ratio features.
5. **Classifies test files** using the same feature pipeline and k-NN logic.
6. **Calculates classification accuracy** and saves results to a CSV file.

### 📈 Use Case:
- Real-time classification of emergency vehicle sirens.
- Works well where spectral band separation is effective (e.g., ambulance vs. firetruck tones).

---

## 🧠 Summary

| File              | Purpose                                                      |
|-------------------|--------------------------------------------------------------|
| `part1_classify.m`| Feature-based audio classifier using MFCC/FFT + distance     |
| `filterDesign.m`  | DSP-based classifier using filtered energy ratios and k-NN   |

Both scripts are fully implemented in MATLAB and intended for audio classification tasks in signal processing coursework or research.

---

