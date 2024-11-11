# src/evaluate.py
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import joblib

def evaluate_model(model_path, input_path, output_path):
    print("#######Model Evalauation Stage Initiated############")
    data = pd.read_csv(input_path)
    X = data[['hours']]
    y = data['marks']

    model = joblib.load(model_path)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    mape = mean_absolute_percentage_error(y,predictions)
    rmse = mean_absolute_error(y,predictions)

    metrics = {'mse': mse,'mape':mape,'rmse':rmse}

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    import sys
    evaluate_model(sys.argv[1], sys.argv[2], sys.argv[3])
