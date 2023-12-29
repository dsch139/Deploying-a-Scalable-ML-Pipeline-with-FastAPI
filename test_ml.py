import pytest
# TODO: add necessary import
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import compute_model_metrics, train_model, inference
from sklearn.ensemble import RandomForestClassifier

# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    test_one checks if the function train_model returns a random forest classifier model.
    """
    # Created data for test
    data = pd.DataFrame(
        {
            'column_1' : [5, 4, 3, 2, 1],
            'column_2' : [0, 1, 0, 1, 0],
            'label' : [0, 1, 0, 1, 0]
        }
    )

    # Process the test data
    X_train, y_train, _,_ = process_data(
        data,
        categorical_features= ['column_2'],
        label= 'label',
        training= True
    )

    #Train model with test data
    random_forest = train_model(X_train, y_train)

    # Check returned model type
    returned_model = RandomForestClassifier
    assert isinstance(random_forest, returned_model)

# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
       Test compute_model_metrics function returns the expected values.
       """
    # Create test data
    y_actual = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])

    # Compute model metrics of test data
    precision, recall, fbeta = compute_model_metrics(y_actual, y_pred)

    # Assign expected values
    expected_precision = 0.6667
    expected_recall = 1.0
    expected_fbeta = 0.8

    assert round(precision, 4) == expected_precision
    assert round(recall, 4) == expected_recall
    assert round(fbeta, 4) == expected_fbeta


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
      Test inference of model
      """
    X_train = np.random.rand(50, 5)
    y_train = np.random.randint(2, size=50)
    random_forest = train_model(X_train, y_train)
    y_preds = inference(random_forest, X_train)

    assert y_train.shape == y_preds.shape

