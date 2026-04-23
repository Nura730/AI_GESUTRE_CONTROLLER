"""
=============================================================================
 TRAINING MODULE - Real-Time Hand Gesture Recognition System
=============================================================================
 Purpose  : Trains a Deep Learning (MLP) model using TensorFlow/Keras on
            hand gesture features extracted from MediaPipe landmarks.
 Author   : Arun S (Nura)
 Project  : Real-Time Hand Gesture Recognition for Device Control using DL
=============================================================================

 Architecture:
   Input Layer  → 57 features (42 coords + 8 distances + 7 angles)
   Dense(128)   → ReLU + BatchNorm + Dropout(0.3)
   Dense(64)    → ReLU + BatchNorm + Dropout(0.3)
   Dense(32)    → ReLU + Dropout(0.2)
   Output(N)    → Softmax (N = number of gesture classes)

 Why Deep Learning over RandomForest:
   1. Learns non-linear feature interactions automatically
   2. Softmax output gives calibrated confidence probabilities
   3. Scales better with more data and gesture classes
   4. Generalizes better to unseen hand orientations
   5. Batch normalization handles feature scale variations
   6. Dropout prevents overfitting on small datasets
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# TensorFlow / Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Sklearn imports for evaluation & preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import warnings
warnings.filterwarnings('ignore')


# ========================= CONFIGURATION =========================

CSV_FILE = "gestures.csv"
MODEL_FILE = "gesture_model.h5"
LABEL_ENCODER_FILE = "label_encoder.json"
SCALER_FILE = "scaler.json"
ACTION_MAP_FILE = "gesture_action_map.json"

# Training Hyperparameters
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.15
RANDOM_STATE = 42


# ========================= DATA LOADING =========================

def load_and_prepare_data(csv_path):
    """
    Loads gesture data from CSV and prepares features/labels.
    
    Handles both old format (43 features) and new format (57 features)
    by detecting the number of numeric columns automatically.
    
    Returns:
        X: Feature matrix (numpy array)
        y: Gesture labels (numpy array of strings)
        gesture_action_map: Dict mapping gesture names to actions
    """
    print("[*] Loading dataset from:", csv_path)
    
    # Load CSV
    df = pd.read_csv(csv_path, header=None)
    
    # Detect if first row is a header (contains text in first column)
    try:
        float(df.iloc[0, 0])
    except (ValueError, TypeError):
        print("   [i] Detected header row, skipping...")
        df = df.iloc[1:]
        df.reset_index(drop=True, inplace=True)
    
    print(f"   [*] Total samples: {len(df)}")
    print(f"   [*] Total columns: {df.shape[1]}")
    
    # Last two columns are gesture_name and action
    gesture_col = df.iloc[:, -2]
    action_col = df.iloc[:, -1]
    
    # All other columns are features
    X = df.iloc[:, :-2].values.astype(np.float32)
    y = gesture_col.values
    
    # Build gesture-to-action mapping
    gesture_action_map = {}
    for gesture, action in zip(y, action_col):
        gesture_action_map[gesture] = action
    
    print(f"\n   [*] Gesture Classes Found:")
    for g, a in gesture_action_map.items():
        count = np.sum(y == g)
        print(f"      - {g:15s} -> {a:20s} ({count} samples)")
    
    print(f"\n   [*] Feature dimensions: {X.shape[1]}")
    
    return X, y, gesture_action_map


# ========================= MODEL BUILDING =========================

def build_model(input_dim, num_classes):
    """
    Builds a lightweight MLP (Multi-Layer Perceptron) for gesture classification.
    
    Architecture designed for:
      - Fast inference (< 5ms per prediction)
      - Good generalization on small datasets (< 5000 samples)
      - Calibrated probability outputs via softmax
    
    Args:
        input_dim: Number of input features
        num_classes: Number of gesture classes
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        
        # Layer 1: Feature expansion
        Dense(128, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Layer 2: Feature compression
        Dense(64, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Layer 3: Final feature refinement
        Dense(32, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.2),
        
        # Output layer: Softmax for probability distribution
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile with Adam optimizer
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ========================= TRAINING =========================

def train(X, y, gesture_action_map):
    """
    Full training pipeline:
      1. Encode labels
      2. Scale features
      3. Split data
      4. Train model with callbacks
      5. Evaluate and generate reports
      6. Save model and artifacts
    """
    
    print("\n" + "=" * 60)
    print("  [>] TRAINING PIPELINE")
    print("=" * 60)
    
    # -------- 1. Encode Labels --------
    print("\n[1] Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    y_onehot = to_categorical(y_encoded, num_classes=num_classes)
    
    class_names = label_encoder.classes_.tolist()
    print(f"   Classes: {class_names}")
    print(f"   Num classes: {num_classes}")
    
    # -------- 2. Scale Features --------
    print("\n[2] Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   Mean range: [{X_scaled.mean(axis=0).min():.4f}, {X_scaled.mean(axis=0).max():.4f}]")
    print(f"   Std range:  [{X_scaled.std(axis=0).min():.4f}, {X_scaled.std(axis=0).max():.4f}]")
    
    # -------- 3. Train/Test Split --------
    print(f"\n[3] Splitting data (test={TEST_SPLIT*100:.0f}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_onehot,
        test_size=TEST_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )
    # Also keep original labels for test set evaluation
    _, _, y_train_labels, y_test_labels = train_test_split(
        X_scaled, y_encoded,
        test_size=TEST_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )
    
    print(f"   Training samples:   {len(X_train)}")
    print(f"   Testing samples:    {len(X_test)}")
    
    # -------- 4. Build Model --------
    print(f"\n[4] Building model...")
    input_dim = X_train.shape[1]
    model = build_model(input_dim, num_classes)
    model.summary()
    
    # -------- 5. Training Callbacks --------
    callbacks = [
        # Stop early if no improvement for 20 epochs
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        # Save best model
        ModelCheckpoint(
            MODEL_FILE,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # -------- 6. Train Model --------
    print(f"\n[5] Training for up to {EPOCHS} epochs...")
    print(f"   Batch size:       {BATCH_SIZE}")
    print(f"   Learning rate:    {LEARNING_RATE}")
    print(f"   Validation split: {VALIDATION_SPLIT*100:.0f}%")
    print("-" * 60)
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )
    
    # -------- 7. Evaluate on Test Set --------
    print("\n" + "=" * 60)
    print("  [*] EVALUATION RESULTS")
    print("=" * 60)
    
    # Test accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n   [+] Test Loss:     {test_loss:.4f}")
    print(f"   [+] Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Training accuracy (best epoch)
    best_train_acc = max(history.history['accuracy'])
    best_val_acc = max(history.history['val_accuracy'])
    print(f"   [>] Best Training Accuracy:   {best_train_acc*100:.2f}%")
    print(f"   [>] Best Validation Accuracy: {best_val_acc*100:.2f}%")
    
    # Predictions on test set
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Classification Report
    print("\n" + "-" * 60)
    print("  [*] CLASSIFICATION REPORT")
    print("-" * 60)
    report = classification_report(y_test_labels, y_pred, target_names=class_names)
    print(report)
    
    # Overall metrics
    overall_accuracy = accuracy_score(y_test_labels, y_pred)
    print(f"   [*] Overall Accuracy: {overall_accuracy*100:.2f}%")
    
    # -------- 8. Save Artifacts --------
    print("\n[*] Saving model and artifacts...")
    
    # Save the final model (best version already saved by checkpoint)
    model.save(MODEL_FILE)
    print(f"   [+] Model saved: {MODEL_FILE}")
    
    # Save label encoder mapping
    label_map = {
        "classes": class_names,
        "class_to_index": {name: int(idx) for idx, name in enumerate(class_names)}
    }
    with open(LABEL_ENCODER_FILE, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"   [+] Label encoder saved: {LABEL_ENCODER_FILE}")
    
    # Save scaler parameters
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist()
    }
    with open(SCALER_FILE, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print(f"   [+] Scaler saved: {SCALER_FILE}")
    
    # Save gesture-action mapping
    with open(ACTION_MAP_FILE, 'w') as f:
        json.dump(gesture_action_map, f, indent=2)
    print(f"   [+] Action map saved: {ACTION_MAP_FILE}")
    
    # -------- 9. Summary --------
    print("\n" + "=" * 60)
    print("  [*] TRAINING COMPLETE")
    print("=" * 60)
    print(f"   Model:            {MODEL_FILE}")
    print(f"   Test Accuracy:    {test_accuracy*100:.2f}%")
    print(f"   Classes:          {class_names}")
    print(f"   Features:         {input_dim}")
    print(f"   Total params:     {model.count_params():,}")
    print(f"   Epochs trained:   {len(history.history['loss'])}")
    print("=" * 60)
    
    return model, history


# ========================= MAIN =========================

def main():
    """Main entry point for training pipeline."""
    
    print("\n" + "=" * 60)
    print("  [*] DEEP LEARNING GESTURE RECOGNITION - TRAINING")
    print("  [*] " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(CSV_FILE):
        print(f"\n[!] Error: {CSV_FILE} not found!")
        print("   Run data_collection.py first to collect gesture data.")
        return
    
    # Load data
    X, y, gesture_action_map = load_and_prepare_data(CSV_FILE)
    
    # Check minimum samples
    if len(X) < 50:
        print(f"\n[!] Warning: Only {len(X)} samples found.")
        print("   Recommend at least 100 samples per gesture class.")
    
    # Train model
    model, history = train(X, y, gesture_action_map)
    
    print("\n[*] All done! You can now run predict.py for real-time gesture recognition.\n")


if __name__ == "__main__":
    main()