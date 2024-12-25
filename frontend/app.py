import streamlit as st
import requests
import plotly.graph_objects as go

# URL backend
BACKEND_URL = "http://backend:8000"

# Заголовок приложения
st.title("Обучение моделей")

# Сохранение состояния загруженного файла
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = None

# Шаг 1: Загрузка файла
uploaded_file = st.file_uploader("Загрузите zip файл для обучения", type=["zip"])
if uploaded_file is not None and st.session_state["dataset_name"] is None:
    # Отправляем файл на сервер только один раз
    files = {"file": (uploaded_file.name, uploaded_file.read(), "text/csv")}
    upload_response = requests.post(f"{BACKEND_URL}/upload_dataset", files=files)

    if upload_response.status_code == 200:
        st.success("Файл успешно загружен!")
        st.session_state["dataset_name"] = upload_response.json()["filepath"].split("/")[-1]  # Сохраняем имя файла
    else:
        st.error(f"Ошибка загрузки файла: {upload_response.text}")

# Использование сохранённого имени файла
dataset_name = st.session_state["dataset_name"]
if dataset_name:
    st.write(f"Загруженный файл: {dataset_name}")

# Шаг 2: Выбор модели и параметров
if dataset_name:
    st.write("Выберите модель для обучения:")
    model_type = st.selectbox("Модель", ["SVC", "Logistic Regression"])

    params = {}
    if model_type == "SVC":
        # Гиперпараметры для SVC
        params["C"] = st.slider("C (регуляризация)", 0.001, 100.0, 1.0)
        params["kernel"] = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        params["gamma"] = st.selectbox("Gamma", ["scale", "auto"])
        params["class_weight"] = st.selectbox("Class Weight", [None, "balanced"])
    elif model_type == "Logistic Regression":
        # Гиперпараметры для Logistic Regression
        params["solver"] = st.selectbox("Solver", ["liblinear", "saga", "lbfgs"])
        params["penalty"] = st.selectbox("Penalty", ["l1", "l2", "elasticnet", "none"])
        params["C"] = st.selectbox("C", [0.001, 0.01, 0.1, 1, 10, 100])
        params["class_weight"] = st.selectbox("Class Weight", [None, "balanced"])

        # Параметр l1_ratio отображается только если penalty = elasticnet
        if params["penalty"] == "elasticnet":
            params["l1_ratio"] = st.slider("L1 Ratio", 0.0, 1.0, 0.5)
        else:
            params["l1_ratio"] = None  # Если penalty != elasticnet, устанавливаем None

    # Кнопка для обучения модели
    if st.button("Обучить модель"):
        # Формируем запрос на обучение
        train_data = {
            "model_type": model_type,
            "params": params,
            "dataset_name": dataset_name
        }
        train_response = requests.post(f"{BACKEND_URL}/train_model", json=train_data)

        if train_response.status_code == 200:
            st.success("Модель успешно обучена!")
            result = train_response.json()
            exp_name = result["experiment_name"]
            st.write(f"Имя эксперимента: {exp_name}")

            # Список экспериментов
            exps_resp = requests.get(f"{BACKEND_URL}/experiments")
            if exps_resp.status_code == 200:
                exps = exps_resp.json()["experiments"]
                st.write("Существующие эксперименты:", exps)

                # Сравнение экспериментов
                selected_exps = st.multiselect("Выберите эксперименты для сравнения", exps)
                if st.button("Показать кривые обучения выбранных экспериментов"):
                    curves_resp = requests.get(f"{BACKEND_URL}/experiment_curves",
                                               params=[("names", n) for n in selected_exps])
                    if curves_resp.status_code == 200:
                        curves_data = curves_resp.json()
                        fig = go.Figure()
                        for e in curves_data:
                            train_sizes = curves_data[e]["train_sizes"]
                            train_scores = curves_data[e]["train_scores"]
                            val_scores = curves_data[e]["validation_scores"]

                            # Добавляем графики для каждого эксперимента
                            fig.add_trace(go.Scatter(
                                x=train_sizes,
                                y=train_scores,
                                mode='lines+markers',
                                name=f"{e} (Train)"
                            ))
                            fig.add_trace(go.Scatter(
                                x=train_sizes,
                                y=val_scores,
                                mode='lines+markers',
                                name=f"{e} (Validation)"
                            ))

                        fig.update_layout(
                            title="Сравнение кривых обучения",
                            xaxis_title="Размер тренировочного набора",
                            yaxis_title="Точность (Accuracy)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Ошибка загрузки кривых: {curves_resp.text}")
            else:
                st.error("Ошибка получения списка экспериментов")
        else:
            st.error(f"Ошибка обучения модели: {train_response.text}")
