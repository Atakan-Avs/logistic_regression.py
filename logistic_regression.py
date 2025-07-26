import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'Grade': [65, 80, 75, 50, 90],
    'Study_Hours': [5, 7, 6, 3, 8],
    'Passed': [0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Separate independent and dependent variables
X = df[['Grade', 'Study_Hours']]
Y = df['Passed']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)

# Make predictions
Y_pred = model.predict(X_test_scaled)

# Calculate the accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print('Accuracy:', accuracy)
