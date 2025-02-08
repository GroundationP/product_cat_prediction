from utils_and_constants import data_load, PATH, files, TrainRequest
from load_data import app
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import accuracy_score


@app.post("/train")
def train_model(params: TrainRequest, run_name_entry):
    """ This new version is using MLflow to store tests. To start using:
    1 - Please click on 'Try it out'.
    2 - Select a mlflow run name (Ex: first run)
    3 - Adjust hyperparameters if necessary
    4 - 'execute' to train the model and enjoy!
    Args:
    hyperparameters, test name for mlflow

    Return:
    model performance
    """
    global xgb
    EXPERIMENT_NAME = "Rakuten_XGB"
    mlflow.set_experiment(EXPERIMENT_NAME)
    # Define experiment name, run name and artifact_path name
    run_name = run_name_entry
    # Create a new MLflow Experiment tag
    mlflow.set_experiment_tag('mlflow.note.content',
                              "Training a Gradient Boost")
    train_data = data_load(PATH, files)[0][0]
    # Convert data to a pandas DataFrame
    df = pd.DataFrame.from_dict(train_data['data'])
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    with mlflow.start_run(run_name=run_name):
        # Train XGBoost model
        xgb = XGBClassifier(
            max_depth=params.max_depth,
            learning_rate=params.learning_rate,
            n_estimators=params.n_estimators,
            random_state=params.random_state,
            # use_label_encoder=False,
            eval_metric="mlogloss"
        )
        xgb.fit(X_train, y_train)
        # Evaluate the model
        pred_Xtrain_xgb = xgb.predict(X_train)
        pred_Xtest_xgb = xgb.predict(X_test)
        accuracy_train = accuracy_score(y_train, pred_Xtrain_xgb)
        accuracy_test = accuracy_score(y_test, pred_Xtest_xgb)
        # Log parameters, metrics, and classification report to MLflow
        mlflow.log_param("max_depth", params.max_depth)
        mlflow.log_param("learning_rate", params.learning_rate)
        mlflow.log_param("n_estimators", params.n_estimators)
        mlflow.log_param("test_size", params.test_size)
        mlflow.log_param("random_state", params.random_state)
        mlflow.log_metric("accuracy train", accuracy_train)
        mlflow.log_metric("accuracy test", accuracy_test)
        # Log the model
        mlflow.xgboost.log_model(xgb, "xgboost_classifier_model")
    # save the model to disk
    try:
        filename = 'app/models/best_xgb.pkl'
        joblib.dump(xgb, filename)
    except:
        filename = 'models/best_xgb.pkl'
        joblib.dump(xgb, filename)
    return {"message": "Model trained successfully",
            "accuracy train": accuracy_train,
            "accuracy test": accuracy_test}
