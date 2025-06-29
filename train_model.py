import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Step 1: Load dataset
print("Loading dataset...")
try:
    data = pd.read_csv('parkinsons.data')
    print(f"Data loaded successfully. Shape: {data.shape}")
except FileNotFoundError:
    print("Error: 'parkinsons.data' file not found.")
    exit()

# Step 2: Separate features and labels
X = data.drop(columns=['name', 'status'])
y = data['status']

# Step 3: Print class distribution
print("\nTarget class distribution:")
print(y.value_counts())

# Step 4: Train-test split
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Feature scaling
print("Scaling features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train XGBoost classifier
print("Training XGBoost classifier...")
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Step 7: Model evaluation
print("\nEvaluating the model...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Save model and scaler
print("\nSaving model and scaler...")
with open('parkinsons_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved as 'parkinsons_model.pkl' and 'scaler.pkl'")
