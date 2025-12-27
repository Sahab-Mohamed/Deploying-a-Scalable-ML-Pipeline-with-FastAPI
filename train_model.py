import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    save_model,
    performance_on_categorical_slice,
)

# Paths
DATA_PATH = "data/census.csv"
MODEL_DIR = "model"
SLICE_OUTPUT_PATH = "slice_output.txt"

# Categorical features (from dataset documentation)
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

LABEL = "salary"


def main():
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load data
    data = pd.read_csv(DATA_PATH)

    # 2. Train-test split
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    # 3. Process training data
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=True,
    )

    # 4. Process test data
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # 5. Train model
    model = train_model(X_train, y_train)

    # 6. Save model and encoders
    save_model(model, f"{MODEL_DIR}/model.pkl")
    save_model(encoder, f"{MODEL_DIR}/encoder.pkl")
    save_model(lb, f"{MODEL_DIR}/lb.pkl")

    print("Model saved to model/model.pkl")
    print("Encoder saved to model/encoder.pkl")
    print("Label binarizer saved to model/lb.pkl")

    # 7. Inference on test set
    preds = inference(model, X_test)

    # 8. Compute overall metrics
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")

    # 9. Compute performance on categorical slices
    with open(SLICE_OUTPUT_PATH, "w") as f:
        for feature in CATEGORICAL_FEATURES:
            for value in data[feature].unique():
                precision, recall, fbeta = performance_on_categorical_slice(
                    data=data,
                    column_name=feature,
                    slice_value=value,
                    categorical_features=CATEGORICAL_FEATURES,
                    label=LABEL,
                    encoder=encoder,
                    lb=lb,
                    model=model,
                )

                count = data[data[feature] == value].shape[0]

                f.write(
                    f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}\n"
                )
                f.write(f"{feature}: {value}, Count: {count}\n")

    print("Slice performance saved to slice_output.txt")


if __name__ == "__main__":
    main()
