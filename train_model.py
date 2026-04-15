import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv("gestures.csv", header=None)

# LAST TWO columns:
# [-2] = gesture_name
# [-1] = action

X = data.iloc[:, :-2]
y = data.iloc[:, -2]

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

pickle.dump(model, open("gesture_model.pkl", "wb"))

print("Model trained successfully!")