import streamlit as st
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging
from logger_config import logger

# URL backend
BACKEND_URL = "http://backend:8000"

# Заголовок приложения
st.title("Приложение для анализа сигналов ЭКГ и определения состояния здоровья человека")
logger.info("Приложение запущено")

# Сохранение состояния загруженного файла для обучения
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = None
if "eda_performed" not in st.session_state:
    st.session_state["eda_performed"] = False
if "df_exploded" not in st.session_state:
    st.session_state["df_exploded"] = None
if "df3" not in st.session_state:
    st.session_state["df3"] = None
if "top20_diseases" not in st.session_state:
    st.session_state["top20_diseases"] = None
if "top15_diseases" not in st.session_state:
    st.session_state["top15_diseases"] = None
if "dataset_name_prediction" not in st.session_state:
    st.session_state["dataset_name_prediction"] = None


# Part 1: Загрузка файла
uploaded_file = st.file_uploader("Загрузите ZIP-файл для обучения", type=["zip"])
if uploaded_file is not None and st.session_state["dataset_name"] is None:
    logger.info("Файл загружен пользователем: %s", uploaded_file.name)
    # Отправляем файл на сервер только один раз
    files = {"file": (uploaded_file.name, uploaded_file.read(), "application/zip")}
    upload_response = requests.post(f"{BACKEND_URL}/upload_dataset", files=files)

    if upload_response.status_code == 200:
        logger.info("Файл успешно отправлен на сервер: %s", uploaded_file.name)
        st.success("Файл успешно загружен!")
        st.session_state["dataset_name"] = upload_response.json()["filepath"].split("/")[-1]  # Сохраняем имя файла
    else:
        logger.error("Ошибка загрузки файла на сервер: %s", upload_response.text)
        st.error(f"Ошибка загрузки файла: {upload_response.text}")

# Part 2: EDA
st.divider()
st.subheader("Разведочный анализ данных")
dataset_name = st.session_state["dataset_name"]

# Проверка наличия загруженного файла
if st.session_state["dataset_name"]:
    st.write(f"Загруженный файл: {st.session_state['dataset_name']}")

    # Отображение кнопки только после загрузки файла
    if not st.session_state["eda_performed"]:
        if st.button("Провести EDA"):
            logger.info("Пользователь инициировал выполнение EDA для файла: %s", dataset_name)
            with st.spinner("Обработка данных, пожалуйста, подождите..."):
                response = requests.get(
                    f"{BACKEND_URL}/get_eda_info",
                    data={"dataset_name": st.session_state["dataset_name"].replace(".zip", "")}
                )
                if response.status_code == 200:
                    logger.info("EDA успешно выполнен для файла: %s", dataset_name)
                    result = response.json()

                    # Сохраняем данные в session_state
                    st.session_state["df_exploded"] = pd.DataFrame(result["df_exploded"])
                    st.session_state["df3"] = pd.DataFrame(result["df3"])
                    st.session_state["top20_diseases"] = result["top_diseases"]
                    st.session_state["top15_diseases"] = result["top_2_diseases"]
                    st.session_state["eda_performed"] = True
                    st.success("EDA успешно выполнен!")
                else:
                    logger.error("Ошибка выполнения EDA: %s", response.text)
                    st.error(f"Ошибка: {response.status_code} — {response.text}")
    else:
        logger.info("EDA уже выполнен для файла: %s", dataset_name)
        st.success("EDA уже выполнен.")
else:
    logger.warning("Попытка выполнения EDA без загрузки файла")
    st.info("Пожалуйста, загрузите файл, чтобы произвести анализ.")

# Part 3: Выбор модели и параметров
st.divider()
st.subheader("Обучение модели")
if uploaded_file is not None and st.session_state["dataset_name"] is not None:
    st.write("Выберите модель для обучения:")
    model_type = st.selectbox("Модель", ["SVC", "Logistic Regression"])

    params = {}
    if model_type == "SVC":
        # Гиперпараметры для SVC
        params["C"] = st.slider("C", 0.001, 100.0, 1.0)
        params["kernel"] = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        params["gamma"] = st.selectbox("Gamma", ["scale", "auto"])
        params["class_weight"] = st.selectbox("Class Weight", [None, "balanced"])
    elif model_type == "Logistic Regression":
        # Гиперпараметры для Logistic Regression
        params["solver"] = st.selectbox("Solver", ["liblinear", "saga", "lbfgs"])
        params["penalty"] = st.selectbox("Penalty", ["l1", "l2", "elasticnet", "none"])
        params["C"] = st.selectbox("C", [0.001, 0.01, 0.1, 1, 10, 100])
        params["class_weight"] = st.selectbox("Class Weight", [None, "balanced"])

        # Параметр l1_ratio отображается, только если penalty = elasticnet
        if params["penalty"] == "elasticnet":
            params["l1_ratio"] = st.slider("L1 Ratio", 0.0, 1.0, 0.5)
        else:
            params["l1_ratio"] = None  # Если penalty != elasticnet, устанавливаем None

    # Кнопка для обучения модели
    if st.button("Обучить модель"):
        logger.info("Запуск обучения модели %s с параметрами %s", model_type, params)
        # Формируем запрос на обучение
        train_data = {
            "model_type": model_type,
            "params": params,
            "dataset_name": dataset_name
        }
        train_response = requests.post(f"{BACKEND_URL}/train_model", json=train_data)

        if train_response.status_code == 200:
            logger.info("Модель %s успешно обучена", model_type)
            st.success("Модель успешно обучена")
            result = train_response.json()
            exp_name = result["experiment_name"]
            st.write(f"Имя эксперимента: {exp_name}")
        else:
            logger.error("Ошибка обучения модели: %s", train_response.text)
            st.error(f"Ошибка обучения модели: {train_response.text}")
else:
    logger.warning("Попытка обучения модели без загрузки файла")
    st.info("Пожалуйста, загрузите файл, чтобы обучить модель.")


# Part 4: Список моделей
# Сохранение состояния загруженного файла для прогноза
if "dataset_name_prediction" not in st.session_state:
    st.session_state["dataset_name_prediction"] = None
    logger.info("Инициализировано состояние для хранения имени файла прогноза.")

st.divider()
st.subheader("Прогноз по анализу ЭКГ")
exps_resp = requests.get(f"{BACKEND_URL}/experiments")
if exps_resp.status_code == 200:
    logger.info("Успешно получен список обученных моделей с сервера.")
    if "experiments" in exps_resp.json():
        exps = exps_resp.json()["experiments"]
        logger.info(f"Найдено {len(exps)} обученных моделей.")
        df = pd.DataFrame(exps)
        df.columns = ['Model IDs']
        st.write("Список обученных моделей:")
        st.dataframe(df)
    else:
        logger.warning("Список обученных моделей пуст.")
        st.error("Обученные модели отсутствуют.")

    # Выбор модели
    selected_model = st.selectbox("Выберите модель для предсказания", exps)
    logger.info(f"Выбрана модель {selected_model} для прогноза")

    uploaded_file_prediction = st.file_uploader("Загрузите ZIP-файл для прогноза диагноза", type=["zip"])
    if uploaded_file_prediction is not None and st.session_state["dataset_name_prediction"] is None:
        # Отправляем файл на сервер только один раз
        logger.info(f"Файл для прогноза загружен: {uploaded_file_prediction.name}")
        files = {"file": (uploaded_file_prediction.name, uploaded_file_prediction.read(), "text/csv")}
        st.session_state["dataset_name_prediction"] = uploaded_file_prediction.name
        disease_names = None

        if st.button("Получить прогноз по ЭКГ"):
            logger.info("Начат процесс получения прогноза.")
            # Заглушка для получения результата
            upload_inference_response = requests.post(f"{BACKEND_URL}/upload_inference", json={"file": uploaded_file_prediction, "model_name": selected_model})

            if upload_inference_response.status_code == 200:
                logger.info("Модель %s успешно обучена", selected_model)
                st.success("Модель успешно обучена")
                result = upload_inference_response.json()
                disease_names = result["predicts"]

                logger.info("Процесс обучения завершён.")
                st.info(f"Предсказанный диагноз: {disease_names}. Требуется консультация с врачом.")
            else:
                logger.error("Ошибка обучения модели: %s", upload_inference_response.text)
                st.error(f"Ошибка обучения модели: {upload_inference_response.text}")
else:
    st.error("Ошибка получения списка обученных моделей")
