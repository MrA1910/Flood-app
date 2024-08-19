from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# Example: Replace this with your actual data
X = np.array([
    [0.4, 100.0, 5.0, 10.0],
    [0.3, 150.0, 4.0, 8.0],
    [0.6, 50.0, 6.0, 12.0],
    [0.2, 200.0, 3.0, 9.0],
    # Add more rows with actual data
])
y = np.array([1, 1, 0, 1])  # Replace with your actual labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Save the trained model
joblib.dump(model, 'flood_model.pkl')
