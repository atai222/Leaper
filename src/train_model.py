import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib # For saving the model

# 1. Load the feature-engineered data
try:
    df = pd.read_csv('training_data.csv')
except FileNotFoundError:
    print("Error: 'training_data.csv' not found. Run 'create_features.py' first.")
else:
    # 2. Define features (X) and target (y)
    X = df.drop('label', axis=1)
    y = df['label']

    # 3. Split data into training and testing sets
    # The data is highly imbalanced, so 'stratify' is crucial
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    # 4. Initialize and train the Random Forest model
    # 'class_weight' helps the model pay more attention to the rare '1' class
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # 5. Evaluate the model
    y_pred = model.predict(X_test)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    # Classification report gives precision, recall, f1-score
    print(classification_report(y_test, y_pred))

    # 6. Save the trained model for future use
    joblib.dump(model, 'takeoff_detector_model.pkl')
    print("Model saved to 'takeoff_detector_model.pkl'")