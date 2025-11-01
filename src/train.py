# src/train.py
import argparse
import numpy as np
from src.data_utils import load_data, prepare_features, train_test_split_df, save_scaler
from src.model import gradient_descent, save_model, train_sklearn
import joblib

def main(args):
    df = load_data(args.data)
    train_df, val_df = train_test_split_df(df, test_size=0.2)
    X_train, y_train, scaler = prepare_features(train_df, scaler=None, fit_scaler=True)
    X_val, y_val, _ = prepare_features(val_df, scaler=scaler, fit_scaler=False)

    # init weights
    w_init = np.zeros(X_train.shape[1])
    w_final, J_hist = gradient_descent(X_train, y_train, w_init, alpha=args.alpha, iterations=args.iter)

    print("Training finished.")
    print("Weights:", w_final)
    print("Final cost:", J_hist[-1])

    # save manual model (weights) and scaler
    save_model({"w": w_final, "scaler": scaler}, args.out_model)

    # also train sklearn for comparison (on un-biased scaled features)
    X_train_unbiased = X_train[:, 1:]  # remove bias col
    lr = train_sklearn(X_train_unbiased, y_train)
    save_model({"sklearn_model": lr, "scaler": scaler}, args.out_sklearn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path CSV data")
    parser.add_argument("--out_model", default="models/manual_model.joblib")
    parser.add_argument("--out_sklearn", default="models/sklearn_model.joblib")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--iter", type=int, default=2000)
    args = parser.parse_args()
    main(args)
