import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def create_split_csv(csv_path):
    """
    Create train test validation split for tabular train data
    :param csv_path: path to the csv
    """
    df_tabular = pd.read_csv(csv_path)
    column_names = df_tabular.columns
    df_tabular = pd.read_csv(csv_path).to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(df_tabular[:, 1:], df_tabular[:, 0], test_size=0.05,
                                                        random_state=42, stratify=df_tabular[:, 0])
    train_array = np.concatenate((np.expand_dims(y_train, axis=1), x_train), axis=1)
    test_array = np.concatenate((np.expand_dims(y_test, axis=1), x_test), axis=1)

    pd.DataFrame(test_array).to_csv("tabular_test.csv", header=column_names,
                                    index=False)

    x_train, x_valid, y_train, y_valid = train_test_split(train_array[:, 1:], train_array[:, 0],
                                                          test_size=0.2105263157894737,
                                                          random_state=42, stratify=train_array[:, 0])

    train_array = np.concatenate((np.expand_dims(y_train, axis=1), x_train), axis=1)
    valid_array = np.concatenate((np.expand_dims(y_valid, axis=1), x_valid), axis=1)

    pd.DataFrame(train_array).to_csv("tabular_train.csv", header=column_names,
                                     index=False)
    pd.DataFrame(valid_array).to_csv("tabular_valid.csv", header=column_names,
                                     index=False)


class TabularDataset(Dataset):
    def __init__(self, tabular, labels, cls_to_indx):
        """
        Tabular dataset reader for nn training
        :param tabular: datapoints
        :param labels: labels
        :param cls_to_indx: encoding
        """
        self.tabular = tabular
        self.labels = labels
        self.cls_to_idx = cls_to_indx
        self.labels = np.array([self.cls_to_idx[label] for label in labels], dtype=np.int64)

    def __getitem__(self, index):
        tabular_data = self.tabular[index]
        tabular_data = torch.from_numpy(tabular_data).float()
        label = self.labels[index % len(self.labels)]
        return tabular_data, label

    def __len__(self):
        return len(self.labels)


CLASS_NAME_DICT_4 = {
    211: 'pion',
    321: 'kaon',
    2212: 'proton',
    -11: 'positron'
}

if __name__ == '__main__':
    print('Data main')
