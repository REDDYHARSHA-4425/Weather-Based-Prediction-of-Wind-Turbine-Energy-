import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("dataset.csv")

# Remove missing values (important for large datasets)
data = data.dropna()

# Print column names to confirm
print("Columns in dataset:", data.columns)

# Define independent and dependent variables
X = data[['Wind Speed', 'Theoretical_Power_Curve']]
y = data['Active_Power']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print accuracy
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
pickle.dump(model, open("model.sav", "wb"))

print("Model saved successfully!")
