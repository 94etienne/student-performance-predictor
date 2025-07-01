import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Step 1: Generate synthetic dataset
print("Generating synthetic dataset...")
np.random.seed(42)
n = 300

df = pd.DataFrame({
    'study_hours': np.random.normal(5, 2, n).clip(0, 10),
    'attendance_rate': np.random.normal(80, 10, n).clip(40, 100),
    'participation': np.random.randint(0, 2, n),  # 0 or 1
    'assignments_submitted': np.random.randint(5, 11, n),  # 5 to 10
    'parental_support': np.random.choice(['low', 'medium', 'high'], size=n, p=[0.2, 0.5, 0.3])
})

# Create a more realistic target variable
def calculate_performance(row):
    score = 0
    # Study hours contribution (0-30 points)
    score += min(row['study_hours'] * 3, 30)
    # Attendance contribution (0-30 points)  
    score += min(row['attendance_rate'] * 0.3, 30)
    # Participation contribution (0-20 points)
    score += row['participation'] * 20
    # Assignments contribution (0-15 points)
    score += min(row['assignments_submitted'] * 1.5, 15)
    # Parental support contribution (0-5 points)
    if row['parental_support'] == 'high':
        score += 5
    elif row['parental_support'] == 'medium':
        score += 3
    else:
        score += 0
    
    # Add some randomness
    score += np.random.normal(0, 5)
    
    # Pass threshold: 60 points
    return 1 if score >= 60 else 0

df['performance'] = df.apply(calculate_performance, axis=1)

print(f"Dataset created with {len(df)} samples")
print(f"Pass rate: {df['performance'].mean():.2%}")

# Step 2: Prepare features
print("\nPreparing features...")
# Convert categorical feature using get_dummies
df_encoded = pd.get_dummies(df, columns=['parental_support'], drop_first=True)

# Separate features and target
X = df_encoded.drop('performance', axis=1)
y = df_encoded['performance']

print(f"Feature columns: {list(X.columns)}")

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Step 4: Train model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)

model.fit(X_train, y_train)

# Step 5: Evaluate model
print("\nEvaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Step 6: Save model and feature names
print("\nSaving model and feature names...")

# Save the model
with open('student_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save column names (feature names)
pd.DataFrame(X.columns).to_csv("columns.csv", header=False, index=False)

print("Model saved as 'student_model.pkl'")
print("Feature names saved as 'columns.csv'")
print("\nTraining completed successfully!")