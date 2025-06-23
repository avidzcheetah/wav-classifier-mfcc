# wav-classifier-mfcc-fft

This project implements a MATLAB-based audio classification system using signal processing and feature extraction techniques. It includes two main parts:

---

## ✅ Part 1: Audio Classification Using MFCC
### 🎧 `classify_audio.m` – MFCC-Based Audio Classifier

This script classifies audio files from three folders: `class_1`, `class_2`, and `unknown`.

### 🔍 What it does:
1. **Reads audio files** from `class_1`, `class_2`, and `unknown`.
2. **Extracts MFCC features** using `extract_mfcc_fixed.m`.
3. **Computes average Euclidean distance** between the unknown sample and each class's training features.
4. **Assigns each unknown file** to the class with the smallest average distance.
5. **Prints and saves results** to `part1_results.csv` using `cell2csv.m`.

### 📁 Supporting Files:
- `extract_mfcc_fixed.m`: Computes 13 fixed-size MFCC features per file.
- `cell2csv.m`: Converts classification results to a CSV format.

### 📈 Use Case:
- General-purpose sound classification using perceptually relevant MFCC features.
- Suitable for distinguishing between classes where spectral patterns differ.

---

## ✅ Part 2: Filter-Based Audio Classification
### 🚨 `filterDesign.m` – Filter-Based Emergency Sound Classifier

This script classifies emergency vehicle audio (ambulance vs. firetruck) using filter-based energy features and k-NN classification.

### 🔍 What it does:
1. **Reads training data** from `train/ambulance` and `train/firetruck`.
2. **Analyzes frequency content** using FFT to identify class-specific frequency bands.
3. **Applies 3 bandpass filters**:
   - bp1: 600–1100 Hz (ambulance)
   - bp2: 1300–1800 Hz (firetruck)
   - bp3: 2100–2800 Hz (firetruck)
4. **Extracts features** by calculating:
   - **Energy in each filtered signal**
   - **Energy ratios**: E1/E2 and E1/E3
5. **Trains a k-NN classifier** (`k=3`) using energy ratio features.
6. **Tests the model** on unseen data and outputs predictions.
7. **Saves results** to `classification_results_part2_unique.csv`.

### 📊 Plots included:
- **Frequency spectra** of ambulance and firetruck signals
- **Bandpass filter responses**
- **Feature scatter plot** showing class separation
- **Filtered signal output** for visual inspection of each band

### 📈 Use Case:
- Designed to recognize emergency vehicle sirens in real time.
- Effective in scenarios with spectral separation between classes.

---

## 🧠 Summary of Key Files

| File Name             | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `classify_audio.m`    | MFCC-based classifier for `unknown/` audio using distance metrics           |
| `extract_mfcc_fixed.m`| Extracts 13 MFCC features with padding/trimming for fixed length            |
| `cell2csv.m`          | Writes classification results to `part1_results.csv`                        |
| `filterDesign.m`      | Filter-based DSP classifier using energy ratios and k-NN                    |

---

## 💻 Tools & Requirements

- MATLAB R2021 or later
- Audio Toolbox (for MFCC feature extraction)
- Signal Processing Toolbox (for filter design and FFT analysis)

---

## 📦 Output Files

- `part1_results.csv`: Predictions for unknown audio samples (Part 1)
- `classification_results_part2_unique.csv`: Results for test audio (Part 2)

---
