import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd
from math import hypot
import time
from collections import deque
import screen_brightness_control as sbc

# -------- VOLUME --------
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))

def set_volume(level):
    volume.SetMasterVolumeLevelScalar(level, None)
# -------- LOAD MODEL --------
model = pickle.load(open("gesture_model.pkl", "rb"))

# -------- LOAD MAPPING FROM CSV --------
data = pd.read_csv("gestures.csv", header=None)

gesture_action_map = {}

for _, row in data.iterrows():
    gesture = row.iloc[-2]
    action = row.iloc[-1]
    gesture_action_map[gesture] = action

print("Loaded Controls:", gesture_action_map)

# -------- MEDIAPIPE --------
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

gesture_buffer = deque(maxlen=5)
last_action_time = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    prediction = "None"

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:

            x0, y0 = hand.landmark[0].x, hand.landmark[0].y

            data_input = []
            for lm in hand.landmark:
                data_input.append(lm.x - x0)
                data_input.append(lm.y - y0)

            d1 = hypot(
                hand.landmark[4].x - hand.landmark[8].x,
                hand.landmark[4].y - hand.landmark[8].y
            )

            data_input.append(d1)

            max_val = max([abs(i) for i in data_input]) or 1
            data_input = [i / max_val for i in data_input]

            pred = model.predict([data_input])[0]

            gesture_buffer.append(pred)
            prediction = max(set(gesture_buffer), key=gesture_buffer.count)

            # -------- ACTION --------
            if time.time() - last_action_time > 0.7:

                action = gesture_action_map.get(prediction)

                if action == "brightness_low":
                    sbc.set_brightness(20)

                elif action == "brightness_high":
                    sbc.set_brightness(90)

                elif action == "volume_up":
                    set_volume(0.8)

                elif action == "volume_down":
                    set_volume(0.2)

                last_action_time = time.time()

    cv2.putText(frame, f"Gesture: {prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()