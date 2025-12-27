import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics


@pytest.fixture(scope="module")
def sample_data():
    """Create a small sample dataframe for testing."""
    data = {
        "age": [39, 50, 38, 53],
        "workclass": ["State-gov", "Self-emp-not-inc", "Private", "Private"],
        "education": ["Bachelors", "Bachelors", "HS-grad", "11th"],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial", "Handlers-cleaners", "Handlers-cleaners"],
        "relationship": ["Not-in-family", "Husband", "Not-in-family", "Husband"],
        "race": ["White", "White", "White", "Black"],
        "sex": ["Male", "Male", "Male", "Male"],
        "native-country": ["United-States", "United-States", "United-States", "United-States"],
        "salary": ["<=50K", ">50K", "<=50K", ">50K"],
    }
    return pd.DataFrame(data)


def test_process_data_output_shapes(sample_data):
    """Test that process_data returns correct output shapes."""
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True,
    )

    assert X.shape[0] == sample_data.shape[0]
    assert y.shape[0] == sample_data.shape[0]
    assert encoder is not None
    assert lb is not None


def test_train_model_returns_model(sample_data):
    """Test that train_model returns a trained RandomForestClassifier."""
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, _, _ = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True,
    )

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


def test_inference_output_type(sample_data):
    """Test that inference returns predictions of correct length."""
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, _, _ = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True,
    )

    model = train_model(X, y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y)


def test_compute_model_metrics_values():
    """Test compute_model_metrics returns valid numeric values."""
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0
