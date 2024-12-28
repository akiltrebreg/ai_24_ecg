import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

import utils
import pandas as pd
import joblib
import uuid
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import logging
import warnings
from datetime import datetime, timedelta
from logger_config import logger

warnings.filterwarnings("ignore")

RAND = 42

"""
log_path = os.path.join("logs", "backend.log")
if not os.path.exists("logs"):
    os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("backend_logger")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
"""

def train_model(model_type: str, params: dict, dataset_name: str):
    data_path = os.path.join("data", dataset_name)
    if not os.path.exists(data_path):
        raise ValueError("Датасет не найден")
    logger.info("Converting_mat_files_to_df...")
    df = utils.make_df_from_mat_files(data_path)
    logger.info("Preprocess is started")
    X_train_std, X_test_std, y_train, y_test, sc, n_train, n_test = utils.preprocess_dataset(df)
    logger.info("Preprocess is finished. Start fitting")
    if model_type == "Logistic Regression":
        model = LogisticRegression(multi_class='ovr', random_state=RAND, max_iter=1000, **params)
    elif model_type == "SVC":
        model = SVC(probability=True, random_state=RAND, **params)
    else:
        raise ValueError(f"Неизвестный тип модели")
    model.fit(X_train_std, y_train)
    logger.info("Fitting is finished. Start predicting")
    y_pred = model.predict(X_test_std)
    logger.info("Predicting is finished. Start count metrics")
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average='weighted')
    }
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_train_std,
        y_train,
        cv=3,
        scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5),
        random_state=RAND
    )

    mean_train_scores = np.mean(train_scores, axis=1).tolist()
    mean_val_scores = np.mean(val_scores, axis=1).tolist()
    formatted_time = (datetime.now() + timedelta(hours=3)).strftime("%H_%M_%d_%m_%Y")
    short_id = str(uuid.uuid4())[:8]
    experiment_id = f"{formatted_time}_{short_id}"
    exp_dir = os.path.join("experiments", experiment_id)
    os.makedirs(exp_dir, exist_ok=True)
    joblib.dump(model, os.path.join(exp_dir, "model.joblib"))
    joblib.dump(sc, os.path.join(exp_dir, "scaler.joblib"))
    pd.DataFrame([metrics]).to_csv(os.path.join(exp_dir, "metrics.csv"), index=False)
    curves = pd.DataFrame({
        "train_sizes": train_sizes.tolist(),
        "train_scores": mean_train_scores,
        "validation_scores": mean_val_scores
    })
    curves.to_csv(os.path.join(exp_dir, "curves.csv"), index=False)
    return experiment_id


def get_inference(folder_path: str, model_name: str):
    print(f"{Path(folder_path).parts[-1]}")
    print(f"{folder_path}")
    df = utils.make_df_from_mat_files(folder_path) #utils.make_df_from_mat_files(Path(folder_path).parts[-1])
    X, _ = utils.preprocess_dataset(df, True)
    print(f"{X}")
    model_path = os.path.join(os.path.join(Path(folder_path).parent.parent, "experiments"), model_name, "model.joblib")
    model = joblib.load(model_path)
    print(f"{model_path}")
    predictions = model.predict(X)
    predictions = predictions.tolist()

    file_path = os.path.abspath(__file__)
    path = Path(file_path).parent
    file_above = path / "snomed-ct.csv"
    df_snomed = pd.read_csv(file_above, sep=',')
    df_snomed = df_snomed.rename(
        columns={'Dx': 'disease_name', 'SNOMED CT Code': 'labels', 'Abbreviation': 'short_disease_name'})
    disease_dict = dict(zip(df_snomed['labels'], df_snomed['disease_name']))
    disease_names = [disease_dict[disease_id] for disease_id in predictions]
    return disease_names



def list_experiments():
    exp_dir = "experiments"
    if not os.path.exists(exp_dir):
        return []
    return [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]


def get_experiment_metrics(name: str):
    exp_dir = os.path.join("experiments", name)
    if not os.path.exists(exp_dir):
        raise ValueError("Эксперимент не найден")
    metrics_path = os.path.join(exp_dir, "metrics.csv")
    df = pd.read_csv(metrics_path)
    return df.to_dict(orient="records")[0]


def get_experiment_curves(names: list):
    curves_data = {}
    for n in names:
        exp_dir = os.path.join("experiments", n)
        if not os.path.exists(exp_dir):
            continue
        curves_path = os.path.join(exp_dir, "curves.csv")
        df = pd.read_csv(curves_path)
        curves_data[n] = {
            "train_sizes": df["train_sizes"].tolist(),
            "train_scores": df["train_scores"].tolist(),
            "validation_scores": df["validation_scores"].tolist()
        }
    return curves_data


def get_eda_info(dataset_name: str):
    return utils.get_eda_info(dataset_name)
