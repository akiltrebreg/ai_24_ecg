import streamlit as st
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# URL backend
BACKEND_URL = "http://backend:8000"

# Заголовок приложения
st.title("Приложение для анализа сигналов ЭКГ и определения состояния здоровья человека")

# Сохранение состояния загруженного файла для обучения
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = None


# Part 1: Загрузка файла
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

# Part 2: EDA
st.divider()
st.subheader("Разведочный анализ данных")
dataset_name = st.session_state["dataset_name"]
if dataset_name:
    st.write(f"Загруженный файл: {dataset_name}")

# Проверяем, есть ли загруженные данные в session_state
if "df_exploded" not in st.session_state:
    st.session_state["df_exploded"] = None
if "df3" not in st.session_state:
    st.session_state["df3"] = None
if "top20_diseases" not in st.session_state:
    st.session_state["top20_diseases"] = None
if "top15_diseases" not in st.session_state:
    st.session_state["top15_diseases"] = None

# Загрузка данных при нажатии кнопки
if st.button("Провести EDA"):
    with st.spinner("Обработка данных, пожалуйста, подождите..."):
        response = requests.get(f"{BACKEND_URL}/get_eda_info", data={"dataset_name": dataset_name.replace(".zip", "")})
        if response.status_code == 200:
            result = response.json()

            df3 = pd.DataFrame(result["df3"])
            st.write(df3.shape)
            df_exploded = pd.DataFrame(result["df_exploded"])
            st.write(df_exploded.shape)
            top20_diseases = result["top_diseases"]
            st.write(top20_diseases)
            top15_diseases = result["top_2_diseases"]
            st.write(top15_diseases)

            st.session_state["df_exploded"] = df_exploded
            st.session_state["df3"] = df3
            st.session_state["top20_diseases"] = top20_diseases
            st.session_state["top15_diseases"] = top15_diseases
        else:
            st.error(f"Ошибка: {response.status_code} — {response.text}")

# Проверяем, загружены ли данные
if st.session_state["df_exploded"] is not None and st.session_state["df3"] is not None\
    and st.session_state["top20_diseases"] is not None and st.session_state["top15_diseases"] is not None:

    df_exploded = st.session_state["df_exploded"]
    df3 = st.session_state["df3"]
    top20_diseases = st.session_state["top20_diseases"]
    top15_diseases = st.session_state["top15_diseases"]

    # Отображение описательной статистики
    st.markdown("Описательная статистика:")
    st.write(df_exploded.describe())

    # Построение гистограммы
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.histplot(df3['len_disease'], kde=True, ax=ax1)
    ax1.set_xlabel('Количество заболеваний')
    ax1.set_ylabel('Количество пациентов')
    ax1.set_title('Гистограмма распределения количества заболеваний у одного пациента')
    st.pyplot(fig1)

    # Группировка данных
    st.write('Количество женщин и мужчин среди пациентов:')
    st.write(df_exploded.groupby('gender').agg({'id': 'nunique'}))

    # Построение ещё одной гистограммы
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df_exploded, x='age', hue='gender', kde=True, ax=ax2)
    ax2.set_xlabel('Возраст')
    ax2.set_ylabel('Количество пациентов')
    ax2.set_title('Гистограмма распределения возраста')
    st.pyplot(fig2)



    # Интерфейс для выбора заболевания и пола
    selected_disease = st.selectbox("Выберите заболевание", top20_diseases)
    gender_filter = st.radio("Выберите пол", ["Оба", "Мужчины", "Женщины"])

    # Фильтрация данных
    @st.cache_data
    def filter_data(disease, gender_filter, df):
        df_filtered = df[df.short_disease_name == disease]
        if gender_filter != "Оба":
            gender = "Male" if gender_filter == "Мужчины" else "Female"
            df_filtered = df_filtered[df_filtered.gender == gender]
        return df_filtered

    df_to_plot = filter_data(selected_disease, gender_filter, df_exploded)

    # Построение графика
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df_to_plot, x='age', kde=True, ax=ax3)
    ax3.set_xlabel('Возраст')
    ax3.set_ylabel('Количество пациентов')
    ax3.set_title(f'Гистограмма распределения возраста для {selected_disease} ({gender_filter})')
    st.pyplot(fig3)


    @st.cache_data
    def filter_data_disease(disease, df):
        df_filtered = df[df.short_disease_name == disease]
        return df_filtered

    df_to_plot_gender = filter_data_disease(selected_disease, df_exploded)

    total_men_count = df_exploded[df_exploded.gender == 'Male'].shape[0]
    total_women_count = df_exploded[df_exploded.gender == 'Female'].shape[0]

    men_count = df_to_plot_gender[(df_to_plot_gender.gender == 'Male') & (df_exploded.short_disease_name == selected_disease)].shape[0]
    women_count = df_to_plot_gender[(df_to_plot_gender.gender == 'Female') & (df_exploded.short_disease_name == selected_disease)].shape[0]

    total_count = men_count + women_count

    men_ratio = men_count / total_count
    women_ratio = women_count / total_count

    plt.figure(figsize=(10, 5))
    sns.barplot(x=['Мужчины', 'Женщины'], y=[men_ratio, women_ratio])

    plt.ylim(0, 1)
    plt.ylabel('Доля пациентов')
    plt.xlabel('Пол')
    plt.title(f'Доля мужчин и женщин для {selected_disease}')
    plt.show()
    st.pyplot(plt)

    st.info("**На основе загруженных данных возможен прогноз следующих диагнозов:** " + ", ".join(top15_diseases))
#else:
#    st.markdown(":red-background[Требуется загрузка csv-файла.]")


# Part 3: Выбор модели и параметров
if dataset_name:
    st.divider()
    st.subheader("Обучение модели")
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
            st.success("Модель успешно обучена")
            result = train_response.json()
            exp_name = result["experiment_name"]
            st.write(f"Имя эксперимента: {exp_name}")
        else:
            st.error(f"Ошибка обучения модели: {train_response.text}")


# Part 4: Список моделей
# Сохранение состояния загруженного файла для прогноза
if "dataset_name_prediction" not in st.session_state:
    st.session_state["dataset_name_prediction"] = None
