import os
import logging
from logging.handlers import RotatingFileHandler
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing_extensions import Annotated
from training import train_model, list_experiments, get_experiment_metrics, get_experiment_curves
from utils import ALLOWED_EXTENSIONS
import warnings
warnings.filterwarnings("ignore")

log_path = os.path.join("logs", "backend.log")
if not os.path.exists("logs"):
    os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("backend_logger")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI()

@app.post("/upload_dataset")
async def upload_dataset(file: Annotated[UploadFile, File(...)]):
    filename = file.filename
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        logger.error(f"Попытка загрузить неправильный формат файла: {filename}")
        raise HTTPException(status_code=400, detail="Неверный формат файла. Допустим: csv")
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
    return {"message": "Файл успешно загружен", "filepath": file_path}

class TrainRequest(BaseModel):
    model_type: str
    params: dict
    dataset_name: str

@app.post("/train_model")
def train_model_endpoint(req: TrainRequest):
    try:
        exp_name = train_model(req.model_type, req.params, req.dataset_name)
        logger.info(f"Модель {req.model_type} успешно обучена. Эксперимент: {exp_name}")
        return {"message": "Модель успешно обучена", "experiment_name": exp_name}
    except Exception as e:
        logger.error(f"Ошибка обучения модели: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/experiments")
def get_experiments():
    exps = list_experiments()
    logger.info("Запрошен список экспериментов")
    return {"experiments": exps}

@app.get("/experiment_metrics")
def experiment_metrics(name: Annotated[str, Form(...)]):
    metrics = get_experiment_metrics(name)
    return metrics

@app.get("/experiment_curves")
def experiment_curves(names: Annotated[List[str], Form(...)]):
    curves = get_experiment_curves(names)
    return curves
