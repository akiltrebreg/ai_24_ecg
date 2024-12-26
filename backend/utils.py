import logging
from logging.handlers import RotatingFileHandler
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path
from tsfel import feature_extraction
import numpy as np
import os
from scipy.io import loadmat
import shutil
from logger_config import logger

warnings.filterwarnings("ignore")
RAND = 42
ALLOWED_EXTENSIONS = ['.zip']
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

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

def make_df_from_mat_files(path: str):
    def import_key_data(path):
        signal = []
        id = []
        time = []
        prefix = []
        one = []
        two = []
        three = []
        aVR = []
        aVL = []
        aVF = []
        V1 = []
        V2 = []
        V3 = []
        V4 = []
        V5 = []
        V6 = []
        age = []
        gender = []
        labels = []
        ecg_filenames = []
        r = []
        h = []
        x = []
        for subdir, dirs, files in sorted(os.walk(path)):
            for filename in files:
                filepath = subdir + os.sep + filename
                if filepath.endswith(".mat"):
                    data, header_data = load_challenge_data(filepath)

                    signal.append(data)

                    if len(header_data[0][:-1]) > 0:
                        id.append(header_data[0][0:6])
                    else:
                        id.append(np.nan)

                    if len(header_data[0][:-1]) > 0:
                        time.append(header_data[0][11:-1])
                    else:
                        time.append(np.nan)

                    if len(header_data[1][:-1]) > 0:
                        prefix.append(header_data[1][11:29])
                    else:
                        prefix.append(np.nan)

                    if len(header_data[1][:-1]) > 0:
                        one.append(header_data[1][29:-1].replace(' I', ''))
                    else:
                        one.append(np.nan)

                    if len(header_data[2][:-1]) > 0:
                        two.append(header_data[2][29:-1].replace(' II', ''))
                    else:
                        two.append(np.nan)

                    if len(header_data[3][:-1]) > 0:
                        three.append(header_data[3][29:-1].replace(' III', ''))
                    else:
                        three.append(np.nan)

                    if len(header_data[4][:-1]) > 0:
                        aVR.append(header_data[4][29:-1].replace(' aVR', ''))
                    else:
                        aVR.append(np.nan)

                    if len(header_data[5][:-1]) > 0:
                        aVL.append(header_data[5][29:-1].replace(' aVL', ''))
                    else:
                        aVL.append(np.nan)

                    if len(header_data[6][:-1]) > 0:
                        aVF.append(header_data[6][29:-1].replace(' aVF', ''))
                    else:
                        aVF.append(np.nan)

                    if len(header_data[7][:-1]) > 0:
                        V1.append(header_data[7][29:-1].replace(' V1', ''))
                    else:
                        V1.append(np.nan)

                    if len(header_data[8][:-1]) > 0:
                        V2.append(header_data[8][29:-1].replace(' V2', ''))
                    else:
                        V2.append(np.nan)

                    if len(header_data[9][:-1]) > 0:
                        V3.append(header_data[9][29:-1].replace(' V3', ''))
                    else:
                        V3.append(np.nan)

                    if len(header_data[10][:-1]) > 0:
                        V4.append(header_data[10][29:-1].replace(' V4', ''))
                    else:
                        V4.append(np.nan)

                    if len(header_data[11][:-1]) > 0:
                        V5.append(header_data[11][29:-1].replace(' V5', ''))
                    else:
                        V5.append(np.nan)

                    if len(header_data[12][:-1]) > 0:
                        V6.append(header_data[12][29:-1].replace(' V6', ''))
                    else:
                        V6.append(np.nan)

                    if len(header_data[15][5:-1]) > 0:
                        labels.append(header_data[15][5:-1])
                    else:
                        labels.append(np.nan)

                    ecg_filenames.append(filepath)

                    if len(header_data[14][6:-1]) > 0:
                        gender.append(header_data[14][6:-1])
                    else:
                        gender.append(np.nan)

                    if len(header_data[13][6:-1]) > 0:
                        age.append(header_data[13][6:-1])
                    else:
                        age.append(np.nan)

                    if len(header_data[16][:-1].split(' ')[1]) > 0:
                        r.append(header_data[16][:-1].split(' ')[1])
                    else:
                        r.append(np.nan)

                    if len(header_data[17][:-1].split(' ')[1]) > 0:
                        h.append(header_data[17][:-1].split(' ')[1])
                    else:
                        h.append(np.nan)

                    if len(header_data[18][:-1].split(' ')[1]) > 0:
                        x.append(header_data[18][:-1].split(' ')[1])
                    else:
                        x.append(np.nan)

        return signal, id, time, prefix, one, two, three, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6, gender, age, labels, ecg_filenames, r, h, x

    def load_challenge_data(filename):
        x = loadmat(filename)
        data = np.asarray(x['val'], dtype=np.float64)
        new_file = filename.replace('.mat', '.hea')
        input_header_file = os.path.join(new_file)
        with open(input_header_file, 'r') as f:
            header_data = f.readlines()
        return data, header_data

    path = os.path.join(DATA_PATH, path)
    folder_path = os.path.splitext(path)[0]
    signal, id, time, prefix, one, two, three, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6, gender, age, labels, ecg_filenames, r, h, x = import_key_data(
        folder_path)

    df = pd.DataFrame(
        {
            'id': id,
            'time': time,
            'prefix': prefix,
            'one': one,
            'two': two,
            'three': three,
            'aVR': aVR,
            'aVL': aVL,
            'aVF': aVF,
            'V1': V1,
            'V2': V2,
            'V3': V3,
            'V4': V4,
            'V5': V5,
            'V6': V6,
            'gender': gender,
            'age': age,
            'labels': labels,
            'signal': signal,
            'ecg_filename': ecg_filenames,
            'r': r,
            'h': h,
            'x': x
        }
    )
    return df


def create_eda(folder_path: str):
    eda_path = folder_path.replace(".zip", "") + "_eda"
    if os.path.exists(eda_path):
        shutil.rmtree(eda_path)
    os.mkdir(eda_path)
    df = make_df_from_mat_files(folder_path)
    df2 = df.drop(['time', 'prefix'], axis=1)
    col_names = ['one', 'two', 'three', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    for col_name in col_names:
        info_first = []
        for el in df[col_name]:
            first_value, second_value, third_value = map(int, el.split())
            info_first.append(first_value)
        df2[f'{col_name}'] = info_first
    df2 = df2.drop(['r', 'h', 'x'], axis=1)  # Всегда unknown
    df2['labels'] = df2['labels'].apply(lambda x: list(map(int, x.split(','))))
    df2['labels'] = df2['labels'].apply(lambda x: list(set(x)) if isinstance(x, list) else x)  # удаление дубликатов
    df2 = df2.drop(columns={'ecg_filename'})
    file_path = os.path.abspath(__file__)
    path = Path(file_path).parent
    file_above = path / "snomed-ct.csv"
    df_diseases = pd.read_csv(file_above, sep=',')
    df_diseases_code = df_diseases.rename(
        columns={'Dx': 'disease_name', 'SNOMED CT Code': 'labels', 'Abbreviation': 'short_disease_name'})
    df_exploded = df2.explode('labels', ignore_index=True)
    df_exploded = df_exploded.merge(df_diseases_code, how='left', on='labels')
    df_exploded['age'] = df_exploded['age'].replace({'NaN': 0, np.nan: 0})
    df_exploded.age = df_exploded.age.apply(lambda x: int(x))
    median_value_male = df_exploded[(df_exploded.gender == 'Male') & (df_exploded.age != 0)]['age'].median()
    df_exploded.loc[df_exploded.gender == 'Male', 'age'] = df_exploded.loc[df_exploded.gender == 'Male', 'age'].replace(
        0, median_value_male)
    df_exploded.loc[df_exploded.gender == 'Female', 'age'] = df_exploded.loc[
        df_exploded.gender == 'Female', 'age'].replace(0, median_value_male)
    df_exploded.groupby(['disease_name', 'short_disease_name']).agg({'id': 'nunique'}).sort_values(by='id',
                                                                                                   ascending=False)
    df3 = df2
    df3['len_disease'] = df3['labels'].apply(lambda x: len(x))
    df3.drop('signal', axis=1).to_csv(os.path.join(eda_path, "df3.csv"), index=False)
    df_exploded.drop('signal', axis=1).to_csv(os.path.join(eda_path, "df_exploded.csv"), index=False)

    df = df_exploded
    top = 20
    top_2 = 15
    top_diseases = df.groupby('labels').id.count().sort_values(ascending=False)[:top].reset_index().labels.tolist()
    list_top_diseases = pd.DataFrame({'ListValues': top_diseases})
    list_top_diseases.to_csv(os.path.join(eda_path, "top_diseases.csv"), index=False)
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
    top_2_diseases = df_cropped_2.groupby('labels').one.count().sort_values(ascending=False)[
                     :top_2].reset_index().labels.tolist()
    list_top_2_diseases = pd.DataFrame({'ListValues': top_2_diseases})
    list_top_2_diseases.to_csv(os.path.join(eda_path, "top_2_diseases.csv"), index=False)

def preprocess_dataset(df: pd.DataFrame, dataset_name: str):
    eda_path = os.path.join(DATA_PATH, dataset_name + '_eda')
    if os.path.exists(eda_path):
        shutil.rmtree(eda_path)
    os.mkdir(eda_path)
    df2 = df.drop(['time', 'prefix'], axis=1)
    col_names = ['one', 'two', 'three', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    for col_name in col_names:
        info_first = []
        for el in df[col_name]:
            first_value, second_value, third_value = map(int, el.split())
            info_first.append(first_value)
        df2[f'{col_name}'] = info_first
    df2 = df2.drop(['r', 'h', 'x'], axis=1)  # Всегда unknown
    df2['labels'] = df2['labels'].apply(lambda x: list(map(int, x.split(','))))
    df2['labels'] = df2['labels'].apply(lambda x: list(set(x)) if isinstance(x, list) else x)  # удаление дубликатов
    df2 = df2.drop(columns={'ecg_filename'})
    file_path = os.path.abspath(__file__)
    path = Path(file_path).parent
    file_above = path / "snomed-ct.csv"

    df_diseases = pd.read_csv(file_above, sep=',')
    df_diseases_code = df_diseases.rename(
        columns={'Dx': 'disease_name', 'SNOMED CT Code': 'labels', 'Abbreviation': 'short_disease_name'})
    df_exploded = df2.explode('labels', ignore_index=True)
    df_exploded = df_exploded.merge(df_diseases_code, how='left', on='labels')
    df_exploded['age'] = df_exploded['age'].replace({'NaN': 0, np.nan: 0})
    df_exploded.age = df_exploded.age.apply(lambda x: int(x))
    median_value_male = df_exploded[(df_exploded.gender == 'Male') & (df_exploded.age != 0)]['age'].median()
    df_exploded.loc[df_exploded.gender == 'Male', 'age'] = df_exploded.loc[df_exploded.gender == 'Male', 'age'].replace(
        0, median_value_male)
    median_value_female = df_exploded[(df_exploded.gender == 'Female') & (df_exploded.age != 0)]['age'].median()
    df_exploded.loc[df_exploded.gender == 'Female', 'age'] = df_exploded.loc[
        df_exploded.gender == 'Female', 'age'].replace(0, median_value_male)
    df_exploded.groupby(['disease_name', 'short_disease_name']).agg({'id': 'nunique'}).sort_values(by='id',
                                                                                                   ascending=False)
    ########################################## ВОТ ЭТОТ НУЖЕН ДЛЯ ГРАФИКОВ И df_exploded ############################
    df3 = df2.copy()
    # print(df_exploded.signal[0])
    # print(type(df_exploded.signal[0][0]))
    df3['len_disease'] = df3['labels'].apply(lambda x: len(x))
    # df3.to_hdf(eda_path, key='df3', mode='w')
    df3.drop('signal', axis = 1).to_csv(os.path.join(eda_path, "df3.csv"), index=False)
    df_exploded.drop('signal', axis = 1).to_csv(os.path.join(eda_path, "df_exploded.csv"), index=False)

    # df3.to_parquet(os.path.join(eda_path, "df3.parquet"), engine='pyarrow', compression='snappy')
    # df_exploded.to_hdf(eda_path, key='df_exploded', mode='a')
    # df_exploded.to_parquet(os.path.join(eda_path, "df_exploded.parquet"), engine='pyarrow', compression='snappy')
    for i in range(len(col_names)):
        col_name = col_names[i]
        df_exploded[f'{col_name}_spectral_entropy'] = df_exploded['signal'].apply(
            lambda x: feature_extraction.features.spectral_entropy(x[i], fs=500))
    for col_name in col_names:
        df_exploded[f'{col_name}_spectral_variation'] = df_exploded['signal'].apply(
            lambda x: feature_extraction.features.spectral_variation(x[i], fs=500))

    for col_name in col_names:
        df_exploded[f'{col_name}_mfcc'] = df_exploded['signal'].apply(
            lambda x: feature_extraction.features.mfcc(x[i], fs=500))

    for col_name in col_names:
        df_exploded[f'{col_name}_spectral_decrease'] = df_exploded['signal'].apply(
            lambda x: feature_extraction.features.spectral_decrease(x[i], fs=500))

    for col_name in col_names:
        df_exploded[f'{col_name}_mean_abs_diff'] = df_exploded['signal'].apply(
            lambda x: feature_extraction.features.mean_abs_diff(x[i]))

    for col_name in col_names:
        df_exploded[f'{col_name}_mean_diff'] = df_exploded['signal'].apply(
            lambda x: feature_extraction.features.mean_diff(x[i]))

    for col_name in col_names:
        df_exploded[f'{col_name}_abs_energy'] = df_exploded['signal'].apply(
            lambda x: feature_extraction.features.abs_energy(x[i]))

    for col_name in col_names:
        df_exploded[f'{col_name}_enthropy'] = df_exploded['signal'].apply(
            lambda x: feature_extraction.features.entropy(x[i]))

    for col_name in col_names:
        df_exploded[f'{col_name}_skewness'] = df_exploded['signal'].apply(
            lambda x: feature_extraction.features.skewness(x[i]))

    for col_name in col_names:
        df_exploded[f'{col_name}_kurtosis'] = df_exploded['signal'].apply(
            lambda x: feature_extraction.features.kurtosis(x[i]))
    def split_mfcc_columns(df):
        mfcc_columns = [col for col in df.columns if 'mfcc' in col]
        for col in mfcc_columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].str.strip('()').apply(lambda x: [float(re.search(r"[-+]?\d*\.\d+|\d+", i).group()) for i in x.split(', ')])
            for i in range(len(df[col][0])):
                new_col_name = f"{col}_{i}"
                df[new_col_name] = df[col].apply(lambda x: x[i] if len(x) > i else None)
            df.drop(columns=[col], inplace=True)
        return df

    print(df_exploded)
    df = split_mfcc_columns(df_exploded)
    top = 20
    top_2 = 15
    top_diseases = df.groupby('labels').id.count().sort_values(ascending=False)[:top].reset_index().labels.tolist()
    # top_diseases.to_hdf(eda_path, key='top_diseases', mode='a')
    list_top_diseases = pd.DataFrame({'ListValues': top_diseases})
    # list_top_diseases.to_parquet(os.path.join(eda_path, "top_diseases.parquet"), engine='pyarrow', compression='snappy')

    list_top_diseases.to_csv(os.path.join(eda_path, "top_diseases.csv"), index=False)
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
    # top_2_diseases.to_hdf(eda_path, key='top_2_diseases', mode='a')
    list_top_2_diseases = pd.DataFrame({'ListValues': top_2_diseases})
    list_top_2_diseases.to_csv(os.path.join(eda_path, "top_2_diseases.csv"), index=False)
    # list_top_2_diseases.to_parquet(os.path.join(eda_path, "top_2_diseases.parquet"), engine='pyarrow', compression='snappy')
    X = df_cropped_2[df_cropped_2.labels.isin(top_2_diseases)].drop(['labels', 'signal', 'disease_name', 'short_disease_name'], axis=1)
    y = df_cropped_2[df_cropped_2.labels.isin(top_2_diseases)]['labels']
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.25,
                                                        stratify=y,
                                                        random_state=RAND)
    sc = StandardScaler()
    X_train_std = pd.DataFrame(sc.fit_transform(X_train), index = X_train.index, columns = X_train.columns)
    X_test_std = pd.DataFrame(sc.transform(X_test), index = X_test.index, columns = X_test.columns)
    y_train = y_train.astype('category')
    y_test = y_test.astype('category')
    return X_train_std, X_test_std, y_train, y_test, sc, X_train_std.shape[0], X_test_std.shape[0]


def get_eda_info(dataset_name: str):
    path = os.path.join(DATA_PATH, dataset_name + "_eda")
    df3 = pd.read_csv(path + '/df3.csv')
    df_exploded = pd.read_csv(path + '/df_exploded.csv')
    top_diseases = pd.read_csv(path + '/top_diseases.csv')['ListValues'].tolist()
    top_2_diseases = pd.read_csv(path + '/top_2_diseases.csv')['ListValues'].tolist()
    return [df3, df_exploded, top_diseases, top_2_diseases]
