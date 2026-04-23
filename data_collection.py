"""
=============================================================================
 DATA COLLECTION MODULE - Real-Time Hand Gesture Recognition System
=============================================================================
 Purpose  : Captures hand landmarks via webcam using MediaPipe and saves
            advanced features (relative coords, distances, angles) to CSV.
 Author   : Arun S (Nura)
 Project  : Real-Time Hand Gesture Recognition for Device Control using DL
=============================================================================
"""

import cv2
import mediapipe as mp
import csv
import numpy as np
from math import hypot, atan2, degrees
import os
import time

# ========================= CONFIGURATION =========================

CSV_FILE = "gestures.csv"

# MediaPipe hand landmark indices for key finger tips and joints
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

# Finger pairs for distance features
DISTANCE_PAIRS = [
    (THUMB_TIP, INDEX_TIP),     # Thumb-Index pinch distance
    (THUMB_TIP, MIDDLE_TIP),    # Thumb-Middle distance
    (INDEX_TIP, MIDDLE_TIP),    # Index-Middle spread
    (MIDDLE_TIP, RING_TIP),     # Middle-Ring spread
    (RING_TIP, PINKY_TIP),      # Ring-Pinky spread
    (THUMB_TIP, PINKY_TIP),     # Thumb-Pinky span (hand openness)
    (WRIST, MIDDLE_TIP),        # Palm length (wrist to middle finger)
    (INDEX_TIP, RING_TIP),      # Cross-finger distance
]

# Angle triplets (vertex is the middle point)
ANGLE_TRIPLETS = [
    (THUMB_TIP, WRIST, INDEX_TIP),      # Thumb-Wrist-Index angle
    (INDEX_TIP, WRIST, PINKY_TIP),      # Index-Wrist-Pinky (hand fan angle)
    (THUMB_TIP, THUMB_MCP, WRIST),      # Thumb bend angle
    (INDEX_TIP, INDEX_PIP, INDEX_MCP),  # Index finger bend
    (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),  # Middle finger bend
    (RING_TIP, RING_PIP, RING_MCP),     # Ring finger bend
    (PINKY_TIP, PINKY_PIP, PINKY_MCP),  # Pinky finger bend
]

# Action mapping for gestures
ACTION_MAP = {
    "1": "brightness_low",
    "2": "brightness_high",
    "3": "volume_up",
    "4": "volume_down"
}


# ========================= FEATURE EXTRACTION =========================

def compute_angle(p1, p2, p3):
    """
    Computes the angle at point p2 formed by p1-p2-p3.
    Returns angle in degrees [0, 360).
    
    Args:
        p1, p2, p3: Tuples of (x, y) coordinates
    Returns:
        Angle in degrees
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle


def extract_features(hand_landmarks):
    """
    Extracts an advanced feature vector from 21 MediaPipe hand landmarks.
    
    Features include:
      1. Relative (x, y) coordinates of all 21 landmarks (42 values)
      2. Distance features between key finger pairs (8 values)
      3. Angle features at key joints (7 values)
    
    Total features: 42 + 8 + 7 = 57 features (before normalization)
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object
    Returns:
        Normalized feature list
    """
    landmarks = hand_landmarks.landmark
    
    # ----- 1. Relative Coordinates (42 features) -----
    # Use wrist (landmark 0) as the reference origin
    x0, y0 = landmarks[WRIST].x, landmarks[WRIST].y
    
    relative_coords = []
    for lm in landmarks:
        relative_coords.append(lm.x - x0)
        relative_coords.append(lm.y - y0)
    
    # ----- 2. Distance Features (8 features) -----
    distances = []
    for i, j in DISTANCE_PAIRS:
        d = hypot(
            landmarks[i].x - landmarks[j].x,
            landmarks[i].y - landmarks[j].y
        )
        distances.append(d)
    
    # ----- 3. Angle Features (7 features) -----
    angles = []
    for p1_idx, p2_idx, p3_idx in ANGLE_TRIPLETS:
        p1 = (landmarks[p1_idx].x, landmarks[p1_idx].y)
        p2 = (landmarks[p2_idx].x, landmarks[p2_idx].y)
        p3 = (landmarks[p3_idx].x, landmarks[p3_idx].y)
        angle = compute_angle(p1, p2, p3)
        angles.append(angle / 180.0)  # Normalize angles to [0, 1]
    
    # ----- Combine All Features -----
    feature_vector = relative_coords + distances
    
    # Normalize relative coords + distances by max absolute value
    max_val = max([abs(f) for f in feature_vector]) or 1.0
    feature_vector = [f / max_val for f in feature_vector]
    
    # Append angle features (already normalized to [0, 1])
    feature_vector.extend(angles)
    
    return feature_vector


def get_feature_names():
    """Returns human-readable names for all features (for CSV header / debugging)."""
    names = []
    for i in range(21):
        names.append(f"lm{i}_x")
        names.append(f"lm{i}_y")
    for i, (a, b) in enumerate(DISTANCE_PAIRS):
        names.append(f"dist_{a}_{b}")
    for i, (a, b, c) in enumerate(ANGLE_TRIPLETS):
        names.append(f"angle_{a}_{b}_{c}")
    names.append("gesture")
    names.append("action")
    return names


# ========================= MAIN DATA COLLECTION =========================

def main():
    """Main data collection loop with real-time webcam visualization."""
    
    print("=" * 60)
    print("  HAND GESTURE DATA COLLECTION - Deep Learning Edition")
    print("=" * 60)
    
    # -------- User Input --------
    gesture_name = input("\n[*] Enter gesture name (e.g., Fist, Open, Point, Peace): ").strip()
    
    print("\n[?] Select Action to Map:")
    print("  1 -> Brightness Low")
    print("  2 -> Brightness High")
    print("  3 -> Volume Up")
    print("  4 -> Volume Down")
    
    choice = input("\n[>] Enter choice (1-4): ").strip()
    action = ACTION_MAP.get(choice, "none")
    
    print(f"\n[OK] Recording: '{gesture_name}' -> Action: '{action}'")
    print("[!] Show your gesture to the webcam. Press 'Q' to stop recording.\n")
    
    # -------- MediaPipe Setup --------
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Error: Could not open webcam!")
        return
    
    # Check if CSV exists; if not, write header
    file_exists = os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0
    
    sample_count = 0
    fps_time = time.time()
    fps = 0
    
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        
        # Write header if new file
        if not file_exists:
            header = get_feature_names()
            writer.writerow(header)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - fps_time + 1e-8)
            fps_time = current_time
            
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    # Draw hand landmarks on frame
                    mp_drawing.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                    
                    # Extract features
                    features = extract_features(hand)
                    
                    # Append gesture name and action
                    row = features + [gesture_name, action]
                    writer.writerow(row)
                    sample_count += 1
            
            # -------- Draw UI Overlay --------
            # Dark semi-transparent bar at the top
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (20, 20, 20), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            # Info text
            cv2.putText(frame, f"Gesture: {gesture_name}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
            cv2.putText(frame, f"Action: {action}", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"Samples: {sample_count}", (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Recording indicator
            cv2.circle(frame, (frame.shape[1] - 30, 70), 8, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (frame.shape[1] - 65, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.imshow("Data Collection - Press Q to Stop", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print(f"\n{'=' * 60}")
    print(f"  [OK] Data collection complete!")
    print(f"  [*] Samples collected: {sample_count}")
    print(f"  [*] Saved to: {CSV_FILE}")
    print(f"  [*] Features per sample: 57 (42 coords + 8 distances + 7 angles)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
