import os
import logging
import shutil
from logging.handlers import RotatingFileHandler
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from typing_extensions import Annotated
import utils
from training import train_model, list_experiments, get_experiment_metrics, \
    get_experiment_curves, get_eda_info
from utils import ALLOWED_EXTENSIONS
import warnings
from logger_config import logger
import training
import numpy as np

warnings.filterwarnings("ignore")


app = FastAPI()


@app.post("/upload_dataset")
async def upload_dataset(file: Annotated[UploadFile, File(...)]) -> dict:
    filename = file.filename
    if not any(filename.endswith(ext)
               for ext in ALLOWED_EXTENSIONS):
        logger.error(f"Попытка загрузить неправильный формат файла: "
                     f"{filename}")
        raise HTTPException(status_code=400,
                            detail="Неверный формат файла. Допустим: csv")
    data_dir = os.path.join("data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, filename)
    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        logger.error("Размер файла превышает 200 МБ")
        raise HTTPException(status_code=400, detail="Файл слишком большой")
    with open(file_path, "wb") as f:
        f.write(content)
    logger.info(f"Файл {filename} успешно загружен")

    folder_path = file_path[:file_path.rfind(".")]
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    logger.info(file_path + "|" + folder_path)
    shutil.unpack_archive(file_path, folder_path)
    logger.info("EDA...")
    utils.create_eda(folder_path)
    logger.info("...EDA")
    logger.info(f"Файл {filename} успешно разархивирован")
    return {"message": "Файл успешно загружен", "filepath": file_path}


@app.post("/upload_inference")
async def upload_inference(
        file: Annotated[UploadFile, File(...)],
        model_name: Annotated[str, Form(...)]
) -> dict:
    filename = file.filename
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        logger.error(f"Попытка загрузить неправильный формат файла: "
                     f"{filename}")
        raise HTTPException(status_code=400,
                            detail="Неверный формат файла. Допустим: csv")
    data_dir = os.path.join("data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, filename)
    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        logger.error("Размер файла превышает 200 МБ")
        raise HTTPException(status_code=400, detail="Файл слишком большой")
    with open(file_path, "wb") as f:
        f.write(content)
    logger.info(f"Файл {filename} успешно загружен")
    folder_path = file_path[:file_path.rfind(".")]
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    logger.info(file_path + "|" + folder_path)
    shutil.unpack_archive(file_path, folder_path)
    logger.info(f"Файл {filename} успешно разархивирован")
    predicts = training.get_inference(folder_path, model_name)
    return {"message": "Предсказание успешно завершено", "predicts": predicts}


class TrainRequest(BaseModel):
    model_type: str
    params: dict
    dataset_name: str


@app.post("/train_model")
def train_model_endpoint(req: TrainRequest):
    try:
        exp_name = train_model(req.model_type, req.params, req.dataset_name)
        logger.info(
            f"Модель {req.model_type} успешно обучена. "
            f"Эксперимент: {exp_name}"
        )
        return {
            "message": "Модель успешно обучена",
            "experiment_name": exp_name
        }
    except Exception as e:
        logger.error(f"Ошибка обучения модели: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiments")
def get_experiments() -> dict:
    res = list_experiments()
    result = []
    for elem, details in res.items():
        entry = {"id": elem, "metrics": {}, "params": {}}
        for el in details:
            if isinstance(el, dict):
                entry["metrics"] = el
            elif isinstance(el, LogisticRegression):
                entry["params"]["model"] = "Logistic Regression"
                entry["params"]["solver"] = str(el.solver)
                entry["params"]["penalty"] = str(el.penalty)
                entry["params"]["C"] = str(el.C)
                entry["params"]["class_weight"] = str(el.class_weight)
                if hasattr(el, 'l1_ratio') and el.l1_ratio is not None:
                    entry["params"]["l1_ratio"] = str(el.l1_ratio)
            elif isinstance(el, SVC):
                entry["params"]["model"] = "SVC"
                entry["params"]["C"] = str(el.C)
                entry["params"]["kernel"] = str(el.kernel)
                entry["params"]["gamma"] = str(el.gamma)
                entry["params"]["class_weight"] = str(el.class_weight)

        result.append(entry)

    logger.info("Запрошен список экспериментов")
    return {"experiments": result}


@app.get("/experiment_metrics")
def experiment_metrics(name: Annotated[str, Form(...)]) -> dict:
    metrics = get_experiment_metrics(name)
    return metrics


@app.get("/experiment_curves")
def experiment_curves(names: Annotated[List[str], Form(...)]) -> dict:
    curves = get_experiment_curves(names)
    return curves


@app.get("/get_eda_info")
def get_eda_info(dataset_name: Annotated[str, Form(...)]) -> dict:
    df3, df_exploded, top_diseases, top_2_diseases = training.\
        get_eda_info(dataset_name)
    return {
        "df3": df3.to_dict(orient="records"),  # DataFrame -> сериал. список
        "df_exploded": df_exploded.to_dict(orient="records"),  # Аналогично
        "top_diseases": [
            int(x) if isinstance(x, np.integer) else x
            for x in top_diseases
        ],
        "top_2_diseases": [
            int(x) if isinstance(x, np.integer) else x
            for x in top_2_diseases
        ],
    }
