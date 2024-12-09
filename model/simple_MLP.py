import pandas as pd
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample


def train_balanced_mlp(X_train, y_train, X_test, y_test=None):
    """
    Function to balance the training data by upsampling the minority class,
    train an MLP model with early stopping, and evaluate the performance on the test data.

    Args:
    - X_train (pd.DataFrame): Features of the training dataset.
    - y_train (pd.Series): Target variable for the training dataset.
    - X_test (pd.DataFrame): Features of the test dataset.
    - y_test (pd.Series or None): Target variable for the test dataset (optional).

    Returns:
    - accuracy (float): Accuracy of the trained model on the test data if y_test is provided.
    - classification_report (str): Classification report of the model performance if y_test is provided.
    - predictions (pd.Series): Model predictions on X_test.
    """
    # Combine X_train and y_train to handle class imbalance
    df_train = X_train.copy()
    df_train['Survived'] = y_train

    # Split into majority and minority class
    df_majority = df_train[df_train['Survived'] == 0]
    df_minority = df_train[df_train['Survived'] == 1]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)

    # Combine majority class with upsampled minority class
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Extract the features and target after balancing
    X_train_balanced = df_balanced.drop('Survived', axis=1)
    y_train_balanced = df_balanced['Survived']

    # Initialize the MLP model with early stopping and a learning rate schedule
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),  # Example architecture
        max_iter=500,  # Max epochs
        random_state=42,
        class_weight='balanced',  # Handle class imbalance
        early_stopping=True,  # Enable early stopping
        validation_fraction=0.1,  # Fraction of training data to use for early stopping
        n_iter_no_change=10,  # Number of iterations with no improvement before stopping
        learning_rate='adaptive',  # Adaptive learning rate
        learning_rate_init=0.001  # Initial learning rate
    )

    # Train the MLP model on the balanced data
    mlp.fit(X_train_balanced, y_train_balanced)

    # Make predictions on the test set
    predictions = mlp.predict(X_test)

    # If y_test is available, evaluate the model performance
    if y_test is not None:
        accuracy = accuracy_score(y_test, predictions)
        classification_rep = classification_report(y_test, predictions)
        return accuracy, classification_rep
    else:
        # Return predictions if no labels are available for the test set
        return predictions

def titanic_MLP(X_train, X_test, y_train, y_test):

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2, random_state=42, class_weight='balanced')
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)

    return mlp, y_pred
