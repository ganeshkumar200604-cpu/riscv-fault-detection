import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                             confusion_matrix, accuracy_score)
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("Step 1: Loading processed dataset...")
df = pd.read_csv('data/processed/final_dataset.csv')
print(f"Dataset shape: {df.shape}")

feature_cols = ['cycles','instructions','cpi','sp','ra','exception_flag']
X = df[feature_cols].values
y = df['label_enc'].values
labels = df['label'].values

print("\nStep 2: Splitting into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

print("\nStep 3: Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Training complete!")

print("\nStep 4: Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

le = joblib.load('models/label_encoder.pkl')
class_names = list(le.classes_)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("\nStep 5: Plotting confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title(f'Confusion Matrix - Accuracy: {accuracy*100:.2f}%')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150)
print("Saved: models/confusion_matrix.png")

print("\nStep 6: Feature Importance...")
importances = model.feature_importances_
for feat, imp in sorted(zip(feature_cols, importances),
                        key=lambda x: x[1], reverse=True):
    bar = '#' * int(imp * 50)
    print(f"  {feat:20s} {imp:.4f}  {bar}")

print("\nStep 7: Saving model...")
joblib.dump(model, 'models/random_forest.pkl')
print("Saved: models/random_forest.pkl")
print("\nTraining Complete!")
