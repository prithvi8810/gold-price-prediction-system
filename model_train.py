import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Get file path
script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, "gold_data_set.csv")

# Check if file exists
if not os.path.exists(path):
    print(f" Dataset file '{path}' not found!")
    print("Please run the dataset creation script first.")
    print("Available files in current directory:")
    for file in os.listdir(script_dir):
        if file.endswith('.csv'):
            print(f"  - {file}")
    exit(1)

print(f" Loading dataset from: {path}")

# Load dataset
try:
    df = pd.read_csv(path)
    print(f" Dataset loaded successfully!")
    print(f" Dataset Shape: {df.shape}")
    print(f" Columns: {list(df.columns)}")
    
    # Show sample data
    print("\n First 5 rows:")
    print(df.head())
    
    # Check for required columns
    required_columns = ['SPX', 'USO', 'SLV', 'EUR/USD', 'GLD']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"\n Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        exit(1)
    
    print(f"\nAll required columns found: {required_columns}")
    
except Exception as e:
    print(f" Error loading dataset: {e}")
    exit(1)

# Features & Target
X = df[['SPX', 'USO', 'SLV', 'EUR/USD']]
y = df['GLD']

print(f"\n Features shape: {X.shape}")
print(f" Target shape: {y.shape}")

# Check for missing values
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    print(" Found missing values, cleaning data...")
    # Remove rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    print(f" Cleaned data shape: {X.shape}")

# Train model
print("\nTraining model, please wait...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f" Training set size: {X_train.shape[0]}")
print(f" Test set size: {X_test.shape[0]}")

# Create and train model
model = DecisionTreeRegressor(random_state=42, max_depth=10, min_samples_split=5)
model.fit(X_train, y_train)

print(" Model training completed!")

# Evaluate model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n Model Performance:")
print(f"Training MAE: ${train_mae:.2f}")
print(f"Test MAE: ${test_mae:.2f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n Feature Importance:")
for _, row in feature_importance.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# Save model
model_file = os.path.join(script_dir, "gold_model.pickle")
try:
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    print(f"\n Model saved as '{model_file}'")
except Exception as e:
    print(f"\n Error saving model: {e}")

# Test prediction with sample data
print(f"\n Sample Prediction:")
sample_features = X_test.iloc[0:1]
sample_prediction = model.predict(sample_features)[0]
actual_value = y_test.iloc[0]

print(f"Input features:")
for feature, value in sample_features.iloc[0].items():
    print(f"  {feature}: {value:.4f}")
print(f"Predicted Gold Price: ${sample_prediction:.2f}")
print(f"Actual Gold Price: ${actual_value:.2f}")
print(f"Prediction Error: ${abs(sample_prediction - actual_value):.2f}")

print(f"\n Training completed successfully!")
print(f" Files created:")
print(f"  - Dataset: gold_data_set.csv")
print(f"  - Model: gold_model.pickle")