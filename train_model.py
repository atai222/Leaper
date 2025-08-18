import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
TRAINING_DATA_FILE = 'training_data.csv'
MODEL_OUTPUT_FILE = 'models/distance_predictor_model.pkl'

# --- SCRIPT ---

def train_distance_model():
    """
    Loads the engineered feature data, trains a RandomForestRegressor model
    to predict phase distance, evaluates it, and saves the final model.
    """
    # 1. Load the feature-engineered data
    try:
        df = pd.read_csv(TRAINING_DATA_FILE)
    except FileNotFoundError:
        print(f"ERROR: Training data file not found at '{TRAINING_DATA_FILE}'")
        print("Please run the create_features.py script first.")
        return

    # Drop non-feature columns and handle any potential missing values
    df_clean = df.drop(['video_name', 'phase_type'], axis=1).dropna()
    
    if df_clean.empty:
        print("ERROR: No data available for training after cleaning. Check your CSV files.")
        return

    # 2. Define features (X) and the target variable (y)
    X = df_clean.drop('distance_m', axis=1)
    y = df_clean['distance_m']

    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training model on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # 4. Initialize and train the RandomForestRegressor model
    # n_estimators is the number of trees in the forest.
    # random_state ensures reproducibility.
    model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
    model.fit(X_train, y_train)

    # 5. Evaluate the model's performance on the unseen test data
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.3f} meters")
    print(f"R-squared (R²): {r2:.3f}")
    print(f"Out-of-Bag (OOB) Score: {model.oob_score_:.3f}")
    print("\nInterpretation:")
    print(f" -> On average, the model's predictions are off by about {mae*100:.1f} cm.")
    print(f" -> The model explains {r2:.1%} of the variance in the jump distances.")

    # 6. Save the trained model for future use
    # Ensure the 'models' directory exists
    os.makedirs(os.path.dirname(MODEL_OUTPUT_FILE), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_FILE)
    print(f"\nModel successfully saved to '{MODEL_OUTPUT_FILE}'")

    # 7. Feature Importance Analysis
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title('Feature Importances for Predicting Jump Distance')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('reports/feature_importances.png')
    print("Feature importance plot saved to 'reports/feature_importances.png'")


if __name__ == "__main__":
    import os
    train_distance_model()

