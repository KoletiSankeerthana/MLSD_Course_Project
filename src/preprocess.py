import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def clean_data(df):
    """
    Cleans the input dataframe: replaces missing values, filters records, and removes outliers.
    """
    # Replace '?' with 'Others'
    df['workclass'] = df['workclass'].replace({'?': 'Others'})
    df['occupation'] = df['occupation'].replace({'?': 'Others'})
    
    # Filter specific workclasses
    df = df[df['workclass'] != 'Without-pay']
    df = df[df['workclass'] != 'Never-worked']
    
    # Outlier detection for age (17-75)
    df = df[(df['age'] <= 75) & (df['age'] >= 17)]
    
    # Outlier detection for educational-num (5-16)
    df = df[(df['educational-num'] <= 16) & (df['educational-num'] >= 5)]
    
    # Remove redundant features
    if 'education' in df.columns:
        df = df.drop(columns=['education'])
        
    return df

def encode_data(df, save_encoders=False, encoders_path='models/encoders.pkl'):
    """
    Encodes categorical features using LabelEncoder.
    If save_encoders is True, it saves the encoders for future prediction use.
    """
    categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
    encoders = {}
    
    # Target encoding
    if 'income' in df.columns:
        df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            
    if save_encoders:
        # Create directory if not exists
        os.makedirs(os.path.dirname(encoders_path), exist_ok=True)
        joblib.dump(encoders, encoders_path)
        
    return df, encoders

def preprocess_pipeline(file_path, save_encoders=True):
    """
    Full pipeline to load, clean, and encode data.
    """
    data = pd.read_csv(file_path)
    data = clean_data(data)
    data, _ = encode_data(data, save_encoders=save_encoders)
    return data

if __name__ == "__main__":
    # Test the script
    try:
        processed_df = preprocess_pipeline('data/adult.csv')
        print("Data Preprocessed Successfully!")
        print(processed_df.head())
    except Exception as e:
        print(f"Error during preprocessing: {e}")
