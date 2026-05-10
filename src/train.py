import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Import local preprocessing module
from preprocess import preprocess_pipeline

def train_model(data_path='data/adult.csv', model_path='models/salary_model.pkl'):
    """
    Loads data, preprocesses it, trains the model, and saves the artifacts.
    """

    print("Starting training process...")

    # 1. Preprocess data
    data = preprocess_pipeline(data_path, save_encoders=True)

    # 2. Split features and target
    X = data.drop(columns=['income'])
    y = data['income']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 3. Create Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    # 4. Train
    print("Training RandomForestClassifier...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"Model trained successfully!")
    print(f"Test Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(pipeline, model_path)

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()