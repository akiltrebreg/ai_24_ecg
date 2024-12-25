import os
from logging.handlers import RotatingFileHandler
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
warnings.filterwarnings("ignore")

RAND = 42

log_path = os.path.join("logs", "backend.log")
if not os.path.exists("logs"):
    os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("backend_logger")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def preprocess_dataset(path: str):
    df = pd.read_csv(path)
    top = 20
    top_2 = 15
    top_diseases = df.groupby('labels').id.count().sort_values(ascending=False)[:top].reset_index().labels.tolist()
    df_preprocessed = df.drop(['id'], axis=1)
    df_preprocessed['gender'] = df_preprocessed['gender'].replace({'Female': 1, 'Male': 0})
    df_preprocessed = df_preprocessed.dropna()
    df_final = df_preprocessed[df_preprocessed.labels.isin(top_diseases)]
    df_cropped = df_final[df_final.labels.isin(top_diseases)]
    subset = list(df_cropped.columns)
    subset.remove('labels')
    subset.remove('short_disease_name')
    subset.remove('signal')
    subset.remove('disease_name')
    df_cropped_2 = df_cropped.drop_duplicates(subset=subset, keep='first')
    top_2_diseases = df_cropped_2.groupby('labels').one.count().sort_values(ascending=False)[:top_2].reset_index().labels.tolist()
    X = df_cropped_2[df_cropped_2.labels.isin(top_2_diseases)].drop(['labels', 'signal', 'disease_name', 'short_disease_name'], axis=1)
    y = df_cropped_2[df_cropped_2.labels.isin(top_2_diseases)]['labels']
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.25,
                                                        stratify=y,
                                                        random_state=RAND)
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std, y_train, y_test, sc, X_train_std.shape[0], X_test_std.shape[0]


def train_model(model_type: str, params: dict, dataset_name: str):
    data_path = os.path.join("data", dataset_name)
    if not os.path.exists(data_path):
        raise ValueError("Датасет не найден")
    logger.info("Unzipping...")
    df = utils.make_df_from_mat_files(dataset_name)
    logger.info("Preprocess is started")
    X_train_std, X_test_std, y_train, y_test, sc, n_train, n_test = utils.preprocess_dataset(df, dataset_name)
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
    experiment_id = str(uuid.uuid4())[:8]
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
