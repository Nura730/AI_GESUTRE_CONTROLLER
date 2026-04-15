import cv2
import mediapipe as mp
import csv
from math import hypot

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

gesture_name = input("Enter gesture name: ")

with open("gestures.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:

                # -------- NORMALIZATION --------
                x0, y0 = hand.landmark[0].x, hand.landmark[0].y

                row = []
                for lm in hand.landmark:
                    row.append(lm.x - x0)
                    row.append(lm.y - y0)

                # -------- DISTANCE FEATURE --------
                d1 = hypot(
                    hand.landmark[4].x - hand.landmark[8].x,
                    hand.landmark[4].y - hand.landmark[8].y
                )

                row.append(d1)

                # -------- SCALE NORMALIZATION --------
                max_val = max([abs(i) for i in row]) if max(row) != 0 else 1
                row = [i / max_val for i in row]

                row.append(gesture_name)
                writer.writerow(row)

        cv2.putText(frame, f"Collecting: {gesture_name}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()