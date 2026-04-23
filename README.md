# 🖐️ Real-Time Hand Gesture Recognition for Device Control using Deep Learning

> **College AIML Project** — A real-time system that recognizes hand gestures via webcam and maps them to device control actions (brightness/volume) using a deep learning model.

---

## 📋 Project Overview

| Feature | Details |
|---------|---------|
| **Input** | Live webcam feed |
| **Hand Detection** | MediaPipe (21 landmarks) |
| **Feature Extraction** | Relative coords + distances + angles (57 features) |
| **Model** | Deep Learning MLP (TensorFlow/Keras) |
| **Output** | Gesture class + confidence → Device action |
| **Actions** | Brightness High/Low, Volume Up/Down |

---

## 🏗️ Architecture

```
Webcam → MediaPipe Hand Detection → Feature Extraction (57 features)
    → StandardScaler Normalization → Deep Learning MLP
    → Gesture Prediction + Confidence → Smoothing Buffer
    → Action Trigger (Brightness / Volume)
```

### Model Architecture (MLP)
```
Input (57 features)
  → Dense(128, ReLU) + BatchNorm + Dropout(0.3)
  → Dense(64, ReLU)  + BatchNorm + Dropout(0.3)
  → Dense(32, ReLU)  + Dropout(0.2)
  → Dense(N, Softmax)  ← N gesture classes
```

---

## 📁 Project Structure

```
AI_Gesture_Controller/
├── data_collection.py       # 📷 Collects gesture data via webcam
├── train_model.py           # 🧠 Trains deep learning model
├── predict.py               # 🎮 Real-time prediction & device control
├── migrate_data.py          # 🔄 Inspects & migrates old CSV data
├── gestures.csv             # 📊 Collected gesture dataset
├── gesture_model.h5         # 💾 Trained Keras model
├── label_encoder.json       # 🏷️ Class name mappings
├── scaler.json              # 📏 Feature scaling parameters
├── gesture_action_map.json  # 🗺️ Gesture → Action mapping
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install opencv-python mediapipe tensorflow numpy pandas scikit-learn matplotlib seaborn screen-brightness-control pycaw comtypes
```

### 2. Collect Data (if starting fresh)
```bash
python data_collection.py
```
- Enter gesture name (e.g., `Fist`, `Open`, `Point`, `Peace`)
- Select action mapping (1-4)
- Show gesture to webcam, press **Q** to stop
- Repeat for each gesture (aim for 200+ samples per class)

### 3. Train the Model
```bash
# If using existing old data, first run:
python migrate_data.py

# Then train:
python train_model.py
```

### 4. Run Real-Time Prediction
```bash
python predict.py
```

---

## 🔬 Feature Engineering (57 Features)

### 1. Relative Coordinates (42 features)
All 21 landmark (x, y) positions relative to the wrist (landmark 0):
```
Feature = landmark[i].x - wrist.x, landmark[i].y - wrist.y
```
This makes the features **position-invariant** — the gesture is recognized regardless of where the hand appears in the frame.

### 2. Distance Features (8 features)
Euclidean distances between key finger pairs:
| Pair | Purpose |
|------|---------|
| Thumb tip ↔ Index tip | Pinch detection |
| Thumb tip ↔ Middle tip | Thumb spread |
| Index tip ↔ Middle tip | V-sign / Peace detection |
| Middle tip ↔ Ring tip | Finger grouping |
| Ring tip ↔ Pinky tip | Fist tightness |
| Thumb tip ↔ Pinky tip | Hand openness span |
| Wrist ↔ Middle tip | Palm length |
| Index tip ↔ Ring tip | Cross-finger spread |

### 3. Angle Features (7 features)
Joint angles at key positions:
- Thumb-Wrist-Index angle (hand shape)
- Index-Wrist-Pinky angle (fan angle)
- Individual finger bend angles (5 fingers)

**Why these features help:**
- Distances capture the *shape* of the hand
- Angles capture *how bent* each finger is
- Together, they provide rich discriminative information that's robust to hand size and distance from camera

---

## 🧠 Why Deep Learning over RandomForest?

| Aspect | RandomForest | Deep Learning (MLP) |
|--------|-------------|-------------------|
| **Confidence Scores** | ❌ Unreliable (averaged tree votes) | ✅ Calibrated softmax probabilities |
| **Feature Interactions** | ❌ Limited (axis-aligned splits) | ✅ Learns complex non-linear combinations |
| **Generalization** | ⚠️ Tends to memorize (overfit) | ✅ Dropout + BatchNorm prevent overfitting |
| **Scalability** | ❌ Slow with many trees | ✅ Efficient batch inference |
| **New Gestures** | ❌ Must retrain entire forest | ✅ Can fine-tune last layer |
| **Deployment** | ⚠️ Large pickle files | ✅ Compact .h5 model |

### Key Advantages for This Project:
1. **Softmax probabilities** enable confidence thresholding — only trigger actions when the model is sure
2. **BatchNormalization** handles varying hand sizes and camera distances
3. **Dropout** prevents overfitting on our small dataset
4. **Gradient-based learning** captures subtle finger position differences between similar gestures

---

## 🔄 Gesture Smoothing

The smoothing system uses a **sliding window buffer** (last 7 predictions):

```
Frame 1: Fist (0.92)   ─┐
Frame 2: Fist (0.88)    │
Frame 3: Open (0.45)    │── Buffer Window
Frame 4: Fist (0.91)    │   → Mode: "Fist"
Frame 5: Fist (0.85)    │   → Avg Confidence: 0.89
Frame 6: Fist (0.90)    │
Frame 7: Fist (0.87)   ─┘
```

**How it works:**
1. Each frame's prediction is added to a circular buffer
2. The **mode** (most frequent gesture) in the buffer is selected
3. Average confidence of the winning gesture is calculated
4. Action only triggers if: confidence ≥ threshold AND cooldown elapsed

**Why it's needed:**
- Eliminates prediction flickering between frames
- Handles brief misclassifications during hand transitions
- Provides stable, reliable gesture output for device control
- Makes the demo look professional and polished

---

## 📊 Evaluation Metrics

After training, the system outputs the following metrics to the console:

1. **Classification Report** — Precision, Recall, F1-score per gesture
2. **Confusion Matrix** — Raw prediction distribution vs. true labels
3. **Test Accuracy** — Final model performance on held-out data

---

## ⚙️ Configuration

Key parameters you can tune in each file:

### `predict.py`
```python
CONFIDENCE_THRESHOLD = 0.65    # Min confidence to accept prediction
SMOOTHING_BUFFER_SIZE = 7      # Frames for gesture smoothing
ACTION_COOLDOWN = 0.8          # Seconds between actions
```

### `train_model.py`
```python
EPOCHS = 150                   # Max training epochs
BATCH_SIZE = 32                # Training batch size
LEARNING_RATE = 0.001          # Adam optimizer learning rate
TEST_SPLIT = 0.2               # 20% data for testing
```

---

## 🎯 Supported Gestures

| Gesture | Action | Description |
|---------|--------|-------------|
| Fist ✊ | Brightness Low | Closed fist |
| Open 🖐️ | Brightness High | All fingers extended |
| Point ☝️ | Volume Up | Index finger pointing |
| Peace ✌️ | Volume Down | Index + middle fingers |

---

## 💡 Tips for Best Results

1. **Data Collection**: Collect 200-300 samples per gesture for best accuracy
2. **Variety**: Move hand slightly during collection (different angles, distances)
3. **Lighting**: Ensure good, even lighting during collection and prediction
4. **Background**: Plain background helps MediaPipe detect hands better
5. **One Hand**: Use only one hand (system tracks single hand)

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **OpenCV** — Video capture & display
- **MediaPipe** — Hand landmark detection
- **TensorFlow/Keras** — Deep learning model
- **scikit-learn** — Data preprocessing & evaluation
- **pycaw** — Windows volume control
- **screen-brightness-control** — Brightness control
- **matplotlib/seaborn** — Visualization

---

## 👨‍💻 Author

**Arun S** — CSE Department

*Real-Time Hand Gesture Recognition System for Device Control using Deep Learning*
