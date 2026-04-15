import cv2
import mediapipe as mp
import csv
from math import hypot

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

# -------- INPUT --------
gesture_name = input("Enter gesture name: ")

print("\nSelect Action:")
print("1 → Brightness Low")
print("2 → Brightness High")
print("3 → Volume Up")
print("4 → Volume Down")

choice = input("Enter choice: ")

action_map = {
    "1": "brightness_low",
    "2": "brightness_high",
    "3": "volume_up",
    "4": "volume_down"
}

action = action_map.get(choice, "none")

print(f"\nRecording Gesture: {gesture_name} → Action: {action}")

with open("gestures.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:

                x0, y0 = hand.landmark[0].x, hand.landmark[0].y

                row = []
                for lm in hand.landmark:
                    row.append(lm.x - x0)
                    row.append(lm.y - y0)

                d1 = hypot(
                    hand.landmark[4].x - hand.landmark[8].x,
                    hand.landmark[4].y - hand.landmark[8].y
                )

                row.append(d1)

                max_val = max([abs(i) for i in row]) or 1
                row = [i / max_val for i in row]

                # 🔥 IMPORTANT: save action also
                row.append(gesture_name)
                row.append(action)

                writer.writerow(row)

        cv2.putText(frame, f"{gesture_name} → {action}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()