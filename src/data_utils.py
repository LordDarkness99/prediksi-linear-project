# src/data_utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(path):
    df = pd.read_csv(path)
    # ensure columns exist
    expected = {"experience", "education", "salary"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"CSV harus memiliki kolom: {expected}")
    return df

def prepare_features(df, scaler=None, fit_scaler=False):
    # gunakan 2 fitur: pengalaman dan pendidikan
    X = df[['experience', 'education']].values
    y = df['salary'].values

    from sklearn.preprocessing import StandardScaler
    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    # tambahkan kolom bias (1) di depan X
    X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

    return X_scaled, y, scaler


def train_test_split_df(df, test_size=0.2, random_state=42):
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)

def save_scaler(scaler, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)
