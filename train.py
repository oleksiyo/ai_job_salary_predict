import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import xgboost as xgb


def load_data(path="./data/ai_job_dataset.csv"):
    df = pd.read_csv(path)

    # Create log-transformed target
    df["target"] = np.log1p(df.salary_usd)
    df = df.drop(columns=["salary_usd"])

    return df


def prepare_data(df):
    # Train/val split
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=11)

    y_train = df_train.target.values
    y_val = df_val.target.values

    df_train = df_train.drop(columns=["target"])
    df_val = df_val.drop(columns=["target"])

    # DictVectorizer
    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(df_train.to_dict(orient="records"))
    X_val = dv.transform(df_val.to_dict(orient="records"))

    return X_train, X_val, y_train, y_val, dv


def train_model(X_train, y_train):
    model = xgb.XGBRegressor(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=11
    )

    model.fit(X_train, y_train)
    return model


def evaluate(model, X_val, y_val):
    y_pred_log = model.predict(X_val)
    y_pred = np.expm1(y_pred_log)
    y_val_real = np.expm1(y_val)
    rmse = np.sqrt(mean_squared_error(y_val_real, y_pred))
    print("Validation RMSE:", rmse)


def save_artifacts(model, dv, model_path="model.bin", dv_path="dv.bin"):
    with open(model_path, "wb") as f_out:
        pickle.dump(model, f_out)

    with open(dv_path, "wb") as f_out:
        pickle.dump(dv, f_out)

    print("Artifacts saved: model.bin, dv.bin")


def main():
    df = load_data()
    X_train, X_val, y_train, y_val, dv = prepare_data(df)
    model = train_model(X_train, y_train)
    evaluate(model, X_val, y_val)
    save_artifacts(model, dv)


if __name__ == "__main__":
    main()