"""
=============================================================================
 PREDICTION MODULE - Real-Time Hand Gesture Recognition System
=============================================================================
 Purpose  : Real-time gesture recognition using trained DL model with
            webcam input, gesture smoothing, confidence thresholding,
            and device control (brightness/volume).
 Author   : Arun S (Nura)
 Project  : Real-Time Hand Gesture Recognition for Device Control using DL
=============================================================================

 Features:
   • Real-time prediction with TensorFlow/Keras model
   • Gesture smoothing via sliding window buffer (reduces flicker)
   • Confidence threshold to filter weak predictions
   • Professional HUD overlay (gesture, confidence, action, FPS)
   • Smooth brightness/volume control
   • Modular and well-documented code
=============================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os
from math import hypot
from collections import deque, Counter

# TensorFlow (silent mode to avoid cluttering the console)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# Device control
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER


# ========================= CONFIGURATION =========================

MODEL_FILE = "gesture_model.h5"
LABEL_ENCODER_FILE = "label_encoder.json"
SCALER_FILE = "scaler.json"
ACTION_MAP_FILE = "gesture_action_map.json"

# Prediction settings
CONFIDENCE_THRESHOLD = 0.65      # Minimum confidence to accept a prediction
SMOOTHING_BUFFER_SIZE = 7        # Number of frames for gesture smoothing
ACTION_COOLDOWN = 0.8            # Seconds between consecutive actions

# MediaPipe landmark indices (must match data_collection.py)
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
THUMB_MCP = 2
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17
INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18

# Distance pairs (must match data_collection.py exactly)
DISTANCE_PAIRS = [
    (THUMB_TIP, INDEX_TIP),
    (THUMB_TIP, MIDDLE_TIP),
    (INDEX_TIP, MIDDLE_TIP),
    (MIDDLE_TIP, RING_TIP),
    (RING_TIP, PINKY_TIP),
    (THUMB_TIP, PINKY_TIP),
    (WRIST, MIDDLE_TIP),
    (INDEX_TIP, RING_TIP),
]

# Angle triplets (must match data_collection.py exactly)
ANGLE_TRIPLETS = [
    (THUMB_TIP, WRIST, INDEX_TIP),
    (INDEX_TIP, WRIST, PINKY_TIP),
    (THUMB_TIP, THUMB_MCP, WRIST),
    (INDEX_TIP, INDEX_PIP, INDEX_MCP),
    (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
    (RING_TIP, RING_PIP, RING_MCP),
    (PINKY_TIP, PINKY_PIP, PINKY_MCP),
]

# HUD Colors (BGR format)
COLOR_BG = (20, 20, 20)
COLOR_GREEN = (0, 255, 128)
COLOR_CYAN = (255, 200, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (180, 180, 180)
COLOR_RED = (60, 60, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_ACCENT = (255, 128, 0)


# ========================= FEATURE EXTRACTION =========================

def compute_angle(p1, p2, p3):
    """Computes the angle at point p2 formed by p1-p2-p3 (degrees)."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def extract_features(hand_landmarks):
    """
    Extracts the same feature vector as data_collection.py.
    Must match exactly for correct predictions.
    
    Returns:
        57-dimensional feature vector (list of floats)
    """
    landmarks = hand_landmarks.landmark
    x0, y0 = landmarks[WRIST].x, landmarks[WRIST].y
    
    # 1. Relative coordinates (42 features)
    relative_coords = []
    for lm in landmarks:
        relative_coords.append(lm.x - x0)
        relative_coords.append(lm.y - y0)
    
    # 2. Distance features (8 features)
    distances = []
    for i, j in DISTANCE_PAIRS:
        d = hypot(landmarks[i].x - landmarks[j].x,
                  landmarks[i].y - landmarks[j].y)
        distances.append(d)
    
    # 3. Angle features (7 features)
    angles = []
    for p1_idx, p2_idx, p3_idx in ANGLE_TRIPLETS:
        p1 = (landmarks[p1_idx].x, landmarks[p1_idx].y)
        p2 = (landmarks[p2_idx].x, landmarks[p2_idx].y)
        p3 = (landmarks[p3_idx].x, landmarks[p3_idx].y)
        angle = compute_angle(p1, p2, p3)
        angles.append(angle / 180.0)
    
    # Normalize coords + distances
    feature_vector = relative_coords + distances
    max_val = max([abs(f) for f in feature_vector]) or 1.0
    feature_vector = [f / max_val for f in feature_vector]
    
    # Append normalized angles
    feature_vector.extend(angles)
    
    return feature_vector


# ========================= LEGACY FEATURE EXTRACTION =========================

def extract_features_legacy(hand_landmarks):
    """
    Extracts features matching the OLD data format (43 features).
    Used when the model was trained on the old CSV format.
    """
    landmarks = hand_landmarks.landmark
    x0, y0 = landmarks[WRIST].x, landmarks[WRIST].y
    
    row = []
    for lm in landmarks:
        row.append(lm.x - x0)
        row.append(lm.y - y0)
    
    d1 = hypot(landmarks[4].x - landmarks[8].x,
               landmarks[4].y - landmarks[8].y)
    row.append(d1)
    
    max_val = max([abs(i) for i in row]) or 1
    row = [i / max_val for i in row]
    
    return row


# ========================= VOLUME CONTROL =========================

def init_volume_control():
    """Initializes Windows volume control via pycaw."""
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None
        )
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        return volume
    except Exception as e:
        print(f"[!] Volume control init failed: {e}")
        return None


def set_volume(volume_interface, level):
    """Sets system volume to a level between 0.0 and 1.0."""
    if volume_interface:
        try:
            volume_interface.SetMasterVolumeLevelScalar(level, None)
        except Exception:
            pass


def set_brightness(level):
    """Sets screen brightness to a percentage (0-100)."""
    try:
        sbc.set_brightness(level)
    except Exception:
        pass


# ========================= HUD DRAWING =========================

def draw_rounded_rect(img, pt1, pt2, color, thickness, radius):
    """Draws a rounded rectangle on the image."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw straight edges
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Draw corners
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_hud(frame, gesture, confidence, action_text, fps, is_active):
    """
    Draws a professional HUD overlay on the video frame.
    
    Displays:
      - Current detected gesture name
      - Confidence percentage with color-coded bar
      - Currently triggered action
      - FPS counter
      - Status indicators
    """
    h, w = frame.shape[:2]
    
    # ---- Top Banner ----
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Title
    cv2.putText(frame, "GESTURE CONTROL", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ACCENT, 2)
    
    # Separator line
    cv2.line(frame, (15, 35), (w - 15, 35), (60, 60, 60), 1)
    
    # Gesture name
    gesture_display = gesture if gesture != "None" else "No Hand Detected"
    gesture_color = COLOR_GREEN if gesture != "None" else COLOR_GRAY
    cv2.putText(frame, f"Gesture: {gesture_display}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
    
    # Confidence bar
    if confidence > 0:
        # Color gradient based on confidence
        if confidence >= 0.85:
            bar_color = COLOR_GREEN
        elif confidence >= 0.65:
            bar_color = COLOR_YELLOW
        else:
            bar_color = COLOR_RED
        
        conf_text = f"Confidence: {confidence * 100:.1f}%"
        cv2.putText(frame, conf_text, (15, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
        
        # Draw confidence bar
        bar_x = 200
        bar_w = 200
        bar_y = 75
        bar_h = 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1)
        filled_w = int(bar_w * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h),
                      bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      COLOR_WHITE, 1)
    
    # Action text
    if action_text and action_text != "None":
        action_display = action_text.replace("_", " ").title()
        cv2.putText(frame, f"Action: {action_display}", (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_CYAN, 2)
    
    # FPS counter (top right)
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_text, (w - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)
    
    # Status indicator (top right)
    status_color = COLOR_GREEN if is_active else (80, 80, 80)
    cv2.circle(frame, (w - 25, 60), 8, status_color, -1)
    cv2.putText(frame, "ACTIVE" if is_active else "IDLE", (w - 100, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
    
    # ---- Bottom Info Bar ----
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 35), (w, h), COLOR_BG, -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, "Press 'Q' to quit | Deep Learning Model | Powered by TensorFlow",
                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)
    
    return frame


# ========================= GESTURE SMOOTHING =========================

class GestureSmoothing:
    """
    Implements gesture smoothing using a sliding window approach.
    
    How it works:
    1. Maintains a buffer of the last N predictions
    2. Returns the most frequent prediction in the buffer (mode)
    3. Only changes the output when a new gesture is consistently detected
    4. This eliminates flickering between frames and provides stable output
    
    Why it's needed:
    - Individual frame predictions can be noisy
    - Hand movements between gestures cause brief misclassifications
    - Users expect stable, non-flickering gesture display
    """
    
    def __init__(self, buffer_size=7):
        self.buffer = deque(maxlen=buffer_size)
        self.current_gesture = "None"
        self.current_confidence = 0.0
    
    def update(self, gesture, confidence):
        """
        Adds a new prediction to the buffer and returns the smoothed result.
        
        Args:
            gesture: Predicted gesture name
            confidence: Prediction confidence (0.0 - 1.0)
        Returns:
            Tuple of (smoothed_gesture, average_confidence)
        """
        self.buffer.append((gesture, confidence))
        
        # Count occurrences of each gesture in the buffer
        gesture_counts = Counter([g for g, c in self.buffer])
        
        # Get the most common gesture
        most_common = gesture_counts.most_common(1)[0]
        smoothed_gesture = most_common[0]
        
        # Calculate average confidence for the smoothed gesture
        relevant_confs = [c for g, c in self.buffer if g == smoothed_gesture]
        avg_confidence = np.mean(relevant_confs)
        
        self.current_gesture = smoothed_gesture
        self.current_confidence = avg_confidence
        
        return smoothed_gesture, avg_confidence
    
    def reset(self):
        """Clears the buffer."""
        self.buffer.clear()
        self.current_gesture = "None"
        self.current_confidence = 0.0


# ========================= MAIN PREDICTION LOOP =========================

def main():
    """Main real-time prediction loop with webcam input."""
    
    print("\n" + "=" * 60)
    print("  [>] REAL-TIME GESTURE RECOGNITION - Deep Learning")
    print("=" * 60)
    
    # -------- 1. Load Model & Artifacts --------
    print("\n[*] Loading model and artifacts...")
    
    # Check required files
    required_files = [MODEL_FILE, LABEL_ENCODER_FILE, SCALER_FILE, ACTION_MAP_FILE]
    for f in required_files:
        if not os.path.exists(f):
            print(f"[!] Error: {f} not found! Run train_model.py first.")
            return
    
    # Load Keras model
    model = tf.keras.models.load_model(MODEL_FILE)
    print(f"   [+] Model loaded: {MODEL_FILE}")
    
    # Load label encoder
    with open(LABEL_ENCODER_FILE, 'r') as f:
        label_map = json.load(f)
    class_names = label_map['classes']
    print(f"   [+] Classes: {class_names}")
    
    # Load scaler
    with open(SCALER_FILE, 'r') as f:
        scaler_params = json.load(f)
    scaler_mean = np.array(scaler_params['mean'], dtype=np.float32)
    scaler_scale = np.array(scaler_params['scale'], dtype=np.float32)
    print(f"   [+] Scaler loaded ({len(scaler_mean)} features)")
    
    # Detect feature dimension to determine old vs new format
    expected_features = len(scaler_mean)
    use_legacy = (expected_features == 43)
    if use_legacy:
        print("   [i] Detected old feature format (43 features)")
    else:
        print(f"   [i] Using advanced feature format ({expected_features} features)")
    
    # Load gesture-action mapping
    with open(ACTION_MAP_FILE, 'r') as f:
        gesture_action_map = json.load(f)
    print(f"   [+] Action map: {gesture_action_map}")
    
    # -------- 2. Initialize Systems --------
    print("\n[*] Initializing systems...")
    
    # Volume control
    volume_ctrl = init_volume_control()
    print(f"   [+] Volume control: {'Ready' if volume_ctrl else 'Unavailable'}")
    
    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    print("   [+] MediaPipe Hands initialized")
    
    # Gesture smoothing
    smoother = GestureSmoothing(buffer_size=SMOOTHING_BUFFER_SIZE)
    print(f"   [+] Gesture smoothing (buffer={SMOOTHING_BUFFER_SIZE})")
    
    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Error: Could not open webcam!")
        return
    
    # Set webcam resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("   [+] Webcam ready (640x480)")
    
    print("\n" + "=" * 60)
    print("  [>] LIVE PREDICTION STARTED - Press 'Q' to quit")
    print("=" * 60 + "\n")
    
    # -------- 3. Main Loop --------
    last_action_time = 0
    fps_time = time.time()
    fps = 0
    frame_count = 0
    action_text = "None"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Calculate FPS
        current_time = time.time()
        frame_count += 1
        if current_time - fps_time >= 0.5:
            fps = frame_count / (current_time - fps_time)
            frame_count = 0
            fps_time = current_time
        
        # Process hand landmarks
        results = hands.process(rgb)
        
        gesture = "None"
        confidence = 0.0
        is_active = False
        
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                # Draw hand skeleton
                mp_drawing.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                )
                
                # Extract features
                if use_legacy:
                    features = extract_features_legacy(hand)
                else:
                    features = extract_features(hand)
                
                features = np.array(features, dtype=np.float32)
                
                # Scale features using saved scaler parameters
                features_scaled = (features - scaler_mean) / (scaler_scale + 1e-8)
                
                # Predict
                input_data = features_scaled.reshape(1, -1)
                prediction = model.predict(input_data, verbose=0)[0]
                
                pred_class_idx = np.argmax(prediction)
                pred_confidence = prediction[pred_class_idx]
                pred_gesture = class_names[pred_class_idx]
                
                # Apply confidence threshold
                if pred_confidence >= CONFIDENCE_THRESHOLD:
                    gesture, confidence = smoother.update(pred_gesture, pred_confidence)
                else:
                    gesture, confidence = smoother.update("Uncertain", pred_confidence)
                
                # -------- Trigger Action --------
                if (gesture not in ["None", "Uncertain"] and
                    confidence >= CONFIDENCE_THRESHOLD and
                    current_time - last_action_time > ACTION_COOLDOWN):
                    
                    action = gesture_action_map.get(gesture, None)
                    
                    if action:
                        is_active = True
                        action_text = action
                        
                        if action == "brightness_low":
                            set_brightness(20)
                        elif action == "brightness_high":
                            set_brightness(90)
                        elif action == "volume_up":
                            set_volume(volume_ctrl, 0.8)
                        elif action == "volume_down":
                            set_volume(volume_ctrl, 0.2)
                        
                        last_action_time = current_time
        else:
            # No hand detected
            smoother.reset()
            gesture = "None"
            confidence = 0.0
        
        # -------- Draw HUD --------
        frame = draw_hud(frame, gesture, confidence, action_text, fps, is_active)
        
        # Show frame
        cv2.imshow("Gesture Control - Deep Learning", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # -------- Cleanup --------
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n[+] Gesture recognition stopped. Goodbye!")


if __name__ == "__main__":
    main()
