import time

import numpy as np
import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn import preprocessing


def split_label(df):
    return df.drop(["id"], axis=1), df.id


# running multiple under sampling methods
X, y = split_label(pd.read_csv("tabular_train.csv"))
column_names = X.columns
X = X.to_numpy()
y = y.to_numpy().astype(np.int64)
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y)
y = label_encoder.transform(y)

# Multiple different undersampling methods tried out here
enn = EditedNearestNeighbours(n_jobs=8)
start_time = time.time()
X_train_res, y_train_res = enn.fit_resample(X, y)
# using label encoder here
y_train_res = label_encoder.inverse_transform(y_train_res)
time_diff = (time.time() - start_time)
print(f'Time = {time_diff}')
df_cdnn = pd.DataFrame(X_train_res, columns=column_names)
df_cdnn["id"] = y_train_res
df_cdnn.to_csv("D:\\Projects\\Imbalanced\\Tabular_data\\edited_nn.csv", index=False)
