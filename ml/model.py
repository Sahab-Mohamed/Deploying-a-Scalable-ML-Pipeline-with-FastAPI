import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
# TODO: add necessary import

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier and return it.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def compute_model_metrics(y, preds):
    """
    Validates model using precision, recall, and F1.
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def inference(model, X):
    """
    Run model inference and return the predictions.
    """
    preds = model.predict(X)
    return preds

def save_model(model, path):
    """
    Save model using pickle.
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path):
    """
    Load model (or encoder) from pickle.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

def performance_on_categorical_slice(
    data,
    column_name,
    slice_value,
    categorical_features,
    label,
    encoder,
    lb,
    model,
):
    """
    Compute metrics on a slice of the data for one categorical feature value.
    """
    # Filter rows where column == slice_value
    slice_df = data[data[column_name] == slice_value]

    # Process slice
    X_slice, y_slice, _, _ = process_data(
        slice_df,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Make predictions
    preds = inference(model, X_slice)

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)

    return precision, recall, fbeta