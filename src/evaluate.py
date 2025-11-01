# src/evaluate.py
import argparse
import numpy as np
from src.data_utils import load_data, prepare_features
from src.model import load_model, predict
from sklearn.metrics import mean_squared_error, r2_score

def main(args):
    df = load_data(args.data)
    X, y, scaler = prepare_features(df, scaler=None, fit_scaler=True)  # or load scaler for proper use
    model_dict = load_model(args.model)
    if "w" in model_dict:
        w = model_dict["w"]
        preds = predict(X, w)
        print("Manual model evaluation:")
    elif "sklearn_model" in model_dict:
        lr = model_dict["sklearn_model"]
        X_unbiased = X[:, 1:]
        preds = lr.predict(X_unbiased)
        print("sklearn model evaluation:")
    else:
        raise ValueError("Model file tidak memiliki key yang diharapkan.")

    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"MSE: {mse:.2f}, RMSE: {mse**0.5:.2f}, R2: {r2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    main(args)
