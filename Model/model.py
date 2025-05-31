# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Select only 4 features: temperature, humidity, ph, rainfall
X = data[["temperature", "humidity", "ph", "rainfall"]]  # Features
y = data["label"]  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Save the model to a pickle file
pickle.dump(model, open("model.pkl", "wb"))

# Optional: Evaluate the model
# accuracy = model.score(X_test, y_test)
# print("Accuracy:", accuracy)

# Example usage (optional)
# new_features = [[26.2724184, 52.12739421, 6.758792552, 127.1752928]]  # temperature, humidity, ph, rainfall
# predicted_crop = model.predict(new_features)
# print("Predicted crop:", predicted_crop)
