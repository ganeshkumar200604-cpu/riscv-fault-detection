import pandas as pd
import glob
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

print("Step 1: Loading all CSV files...")
files = glob.glob('data/raw/*.csv')
print(f"Found {len(files)} files: {files}")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"Total rows loaded: {len(df)}")
print(f"Columns: {list(df.columns)}")

print("\nStep 2: Checking for missing values...")
print(df.isnull().sum())

print("\nStep 3: Removing duplicates...")
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicates")

print("\nStep 4: Checking label distribution...")
print(df['label'].value_counts())

print("\nStep 5: Encoding labels...")
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])
print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print("\nStep 6: Selecting features...")
feature_cols = ['cycles', 'instructions', 'cpi', 'sp', 'ra', 'exception_flag']
X = df[feature_cols]
y = df['label_enc']
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

print("\nStep 7: Normalizing features with StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

print("Before scaling (cycles):")
print(f"  Mean: {df['cycles'].mean():.0f}  Std: {df['cycles'].std():.0f}")
print("After scaling (cycles):")
print(f"  Mean: {X_scaled_df['cycles'].mean():.4f}  Std: {X_scaled_df['cycles'].std():.4f}")

print("\nStep 8: Saving processed dataset...")
os.makedirs('data/processed', exist_ok=True)
X_scaled_df['label'] = df['label'].values
X_scaled_df['label_enc'] = y.values
X_scaled_df.to_csv('data/processed/final_dataset.csv', index=False)
print("Saved: data/processed/final_dataset.csv")

print("\nStep 9: Saving scaler and label encoder...")
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
print("Saved: models/scaler.pkl")
print("Saved: models/label_encoder.pkl")

print("\nPreprocessing Complete!")
print(f"Final dataset shape: {X_scaled_df.shape}")
print(f"Classes: {list(le.classes_)}")
