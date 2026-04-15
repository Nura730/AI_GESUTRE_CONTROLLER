import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("gestures.csv", header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save model
pickle.dump(model, open("gesture_model.pkl", "wb"))

print("Model trained successfully!")