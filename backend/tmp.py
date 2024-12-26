import os
import shutil
import uuid
from pathlib import Path

from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import learning_curve
import numpy as np
import utils



path = 'Georgia.zip'
dataset_name = path.replace(".zip", "")
import time
time1 = time.time()
file_path = os.path.join(Path(os.path.abspath(__file__)).parent, "data\\" + path)
folder_path = file_path.replace('.zip', '')
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
shutil.unpack_archive(file_path, folder_path)
print(time.time() - time1)
utils.create_eda(folder_path)
print(time.time() - time1)
input("END")
for el in utils.get_eda_info(dataset_name):
    print(el)
input("ENDEND")
df = utils.make_df_from_mat_files(path)
pd.set_option('display.expand_frame_repr', False)
print(df)

X_train_std, X_test_std, y_train, y_test, sc, n_train, n_test = utils.preprocess_dataset(df, dataset_name)
# print(X_train_std)
print(y_train)
# print(y_train.head())
# print(y_train.dtype)
# print(y_train.isnull().sum())
# print(y_train.unique())

model = LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000, solver = 'saga', penalty='l2', C=1, class_weight='balanced')
model = LogisticRegression(multi_class='ovr', random_state=42, solver = 'saga', class_weight='balanced')
print("start fit...")
model.fit(X_train_std, y_train)
y_pred = model.predict(X_test_std)

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
    random_state=42
)

mean_train_scores = np.mean(train_scores, axis=1).tolist()
mean_val_scores = np.mean(val_scores, axis=1).tolist()
experiment_id = str(uuid.uuid4())[:8]
exp_dir = os.path.join("experiments", experiment_id)
os.makedirs(exp_dir, exist_ok=True)
curves = pd.DataFrame({
    "train_sizes": train_sizes.tolist(),
    "train_scores": mean_train_scores,
    "validation_scores": mean_val_scores
})

print(metrics)
print(curves)

for el in utils.get_eda_info(dataset_name):
    print(el)
