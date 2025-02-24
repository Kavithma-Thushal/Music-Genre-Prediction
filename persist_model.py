from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load dataset
model_data = pd.read_csv('music.csv')

# Convert gender to numeric values (Female -> 0, Male -> 1)
model_data['gender'] = model_data['gender'].map({'Female': 0, 'Male': 1})

# Prepare features and labels
X = model_data.drop(columns=['genre'])  # Features
y = model_data['genre']  # Labels

# Split data into training and testing sets
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Persist the trained model
joblib.dump(model, 'music-persist.joblib')

print("Model persisted successfully")
