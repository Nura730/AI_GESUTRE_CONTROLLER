import cv2
import mediapipe as mp
import pickle
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import time
from collections import deque
from math import hypot
import screen_brightness_control as sbc
import pyautogui

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("gesture_model.pkl", "rb"))

# ------------------ MEDIAPIPE ------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ------------------ CAMERA ------------------
cap = cv2.VideoCapture(0)

# ------------------ UI ------------------
root = tk.Tk()
root.title("AI Gesture Controller PRO")
root.geometry("950x750")
root.configure(bg="#111111")

label = tk.Label(root)
label.pack()

status = tk.Label(root, text="Gesture: None",
                  font=("Segoe UI", 18, "bold"),
                  fg="cyan", bg="#111111")
status.pack(pady=10)

mode_label = tk.Label(root, text="Mode: BRIGHTNESS",
                      font=("Segoe UI", 14),
                      fg="white", bg="#111111")
mode_label.pack()

action_label = tk.Label(root, text="Last Action: None",
                        font=("Segoe UI", 12),
                        fg="yellow", bg="#111111")
action_label.pack(pady=10)

mode = "BRIGHTNESS"

def switch_mode():
    global mode
    mode = "VOLUME" if mode == "BRIGHTNESS" else "BRIGHTNESS"

btn = tk.Button(root, text="Switch Mode",
                font=("Segoe UI", 12),
                bg="cyan", fg="black",
                padx=10, pady=5,
                command=switch_mode)
btn.pack(pady=10)

# ------------------ CONTROL PANEL ------------------
panel = tk.Frame(root, bg="#222222", width=300, height=200)
panel.pack(pady=20)

tk.Label(panel, text="Controls",
         font=("Segoe UI", 14, "bold"),
         fg="cyan", bg="#222222").pack(pady=5)

tk.Label(panel, text="FIST → Low Brightness",
         fg="white", bg="#222222").pack()

tk.Label(panel, text="OPEN → High Brightness",
         fg="white", bg="#222222").pack()

tk.Label(panel, text="PEACE → Volume Up",
         fg="white", bg="#222222").pack()

tk.Label(panel, text="POINT → Volume Down",
         fg="white", bg="#222222").pack()

# ------------------ STABILITY ------------------
gesture_buffer = deque(maxlen=5)
last_action_time = 0

# ------------------ FPS ------------------
pTime = 0

# ------------------ MAIN LOOP ------------------
def update():
    global last_action_time, pTime

    ret, frame = cap.read()
    if not ret:
        root.after(10, update)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    prediction = "None"
    confidence = 0

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:

            # -------- FEATURE ENGINEERING --------
            x0, y0 = hand.landmark[0].x, hand.landmark[0].y

            data = []
            for lm in hand.landmark:
                data.append(lm.x - x0)
                data.append(lm.y - y0)

            d1 = hypot(
                hand.landmark[4].x - hand.landmark[8].x,
                hand.landmark[4].y - hand.landmark[8].y
            )

            data.append(d1)

            max_val = max([abs(i) for i in data]) if max(data) != 0 else 1
            data = [i / max_val for i in data]

            # -------- PREDICTION WITH CONFIDENCE --------
            probs = model.predict_proba([data])[0]
            max_prob = max(probs)
            pred = model.classes_[np.argmax(probs)]

            if max_prob < 0.7:
                pred = "uncertain"

            confidence = max_prob

            # -------- STABILIZATION --------
            gesture_buffer.append(pred)
            prediction = max(set(gesture_buffer), key=gesture_buffer.count)

            # -------- ACTION --------
            action_text = "None"

            if prediction != "uncertain" and time.time() - last_action_time > 0.5:

                if mode == "BRIGHTNESS":
                    if prediction == "fist":
                        sbc.set_brightness(20)
                        action_text = "Brightness Low"

                    elif prediction == "open":
                        sbc.set_brightness(90)
                        action_text = "Brightness High"

                elif mode == "VOLUME":
                    if prediction == "peace":
                        pyautogui.press("volumeup")
                        action_text = "Volume Up"

                    elif prediction == "point":
                        pyautogui.press("volumedown")
                        action_text = "Volume Down"

                last_action_time = time.time()
                action_label.config(text=f"Last Action: {action_text}")

    # ------------------ UI UPDATE ------------------
    status.config(text=f"Gesture: {prediction} ({confidence:.2f})")
    mode_label.config(text=f"Mode: {mode}")

    # ------------------ FPS ------------------
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # ------------------ DISPLAY ------------------
    img = ImageTk.PhotoImage(Image.fromarray(rgb))
    label.imgtk = img
    label.configure(image=img)

    root.after(10, update)

# ------------------ START ------------------
update()
root.mainloop()

# ------------------ CLEANUP ------------------
cap.release()
cv2.destroyAllWindows()