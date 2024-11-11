# src/train.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_model(input_path, model_path):
    print("############Model Traninig Pipeline Is Initiateed#################")
    data = pd.read_csv(input_path)
    X = data[['hours']]
    y = data['marks']

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, model_path)
    print("Model trained.")

if __name__ == "__main__":
    import sys
    train_model(sys.argv[1], sys.argv[2])
