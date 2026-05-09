import pandas as pd
import joblib
import numpy as np

def load_artifacts(model_path='models/salary_model.pkl', encoders_path='models/encoders.pkl'):
    """
    Loads the trained model pipeline and the label encoders.
    """
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    return model, encoders

def make_prediction(input_data, model_path='models/salary_model.pkl', encoders_path='models/encoders.pkl'):
    """
    Takes a dictionary of input features and returns the prediction.
    """
    model, encoders = load_artifacts(model_path, encoders_path)
    
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    
    # Apply encoders to categorical columns
    categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
    
    for col in categorical_cols:
        if col in df.columns:
            le = encoders[col]
            # Handle unseen labels by mapping them to the first class or a default (simplified for this task)
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                # If label is unknown, use the first class as fallback
                df[col] = 0
                
    # Predict
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    
    return int(prediction[0]), float(np.max(probability))

if __name__ == "__main__":
    # Example prediction
    test_input = {
        'age': 39,
        'workclass': 'State-gov',
        'fnlwgt': 77516,
        'educational-num': 13,
        'marital-status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'gender': 'Male',
        'capital-gain': 2174,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States'
    }
    
    result, prob = make_prediction(test_input)
    label = ">50K" if result == 1 else "<=50K"
    print(f"Prediction: {label} (Confidence: {prob:.2f})")
