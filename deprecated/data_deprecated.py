import os
import pickle
import random
from glob import glob

import numpy as np
import pandas as pd
import scipy.sparse
import torch
import tqdm
from torch.utils.data import Dataset
import warnings

# THIS FILE CONTAINS THE DEPRECATED CODE FOR THE NN MODEL TRAININGS AND IMAGE DATA
warnings.warn("This module is deprecated in favour of machine_learning", DeprecationWarning, stacklevel=2)


def load_pickle(root_folder):
    """
    Pickle loader function
    :param root_folder: root folder path
    :return:
    """
    all_files = [y for x in os.walk(root_folder) for y in glob(os.path.join(x[0], '*.pkl'))]

    objects = np.empty((0,))
    labels = np.empty((0,))
    for file_path in tqdm.tqdm(all_files, 'Loading pickle file'):
        pickle_file = open(file_path, "rb")
        while True:
            try:
                object_pickle = pickle.load(pickle_file)
                labels = np.concatenate((labels, object_pickle[1]))
                objects = np.concatenate((objects, object_pickle[0]))
            except EOFError:
                break
        pickle_file.close()

    return objects, labels


def load_npz(root_folder):
    """
    Npz file format loader
    :param root_folder: path to folder
    :return:
    """
    random.seed(69)
    all_files = [y for x in os.walk(root_folder) for y in glob(os.path.join(x[0], '*.npz'))]
    labels = np.empty((0,))
    objects = np.empty((0, 100))
    for npz in all_files:
        sparse_matrix = scipy.sparse.load_npz(npz)
        dense_matrix = sparse_matrix.todense()
        label = int((str(npz).split('\\')[-1].rstrip('.npz')).split('_')[-1])
        label_matrix = np.zeros((sparse_matrix.shape[0],))
        label_matrix[:] = label

        labels = np.concatenate((labels, label_matrix))
        objects = np.concatenate((objects, dense_matrix))
    perm = random.sample([i for i in range(0, objects.shape[0])], objects.shape[0])
    labels = labels[perm].astype(np.int64)
    objects = objects[perm]
    objects = np.asarray(objects)
    return objects, labels


class ParticleDataset(Dataset):
    def __init__(self, imgs, labels, cls_to_indx, cnn_model):

        self.images = imgs
        self.labels = labels
        self.cls_to_idx = cls_to_indx
        self.labels = np.array([self.cls_to_idx[label] for label in labels], dtype=np.int64)
        self.cnn_model = cnn_model

    def __getitem__(self, index):
        img_array = self.images[index]
        img_array = torch.from_numpy(img_array).float()

        if not self.cnn_model:
            img_array = torch.flatten(img_array)
            img_array = torch.squeeze(img_array)
        else:
            if img_array.shape[-1] != 10:
                assert img_array.shape[-1] == 100
                img_array = img_array.view(-1, 10, 10)

        label = self.labels[index % len(self.labels)]
        return img_array, torch.from_numpy(np.array([0])).float(), label

    def __len__(self):
        return len(self.labels)


class ParticleDatasetCombined(Dataset):
    def __init__(self, images, labels, cls_to_index, tabular_size=6):
        self.images = images
        self.labels = labels
        self.cls_to_idx = cls_to_index
        self.labels = np.array([self.cls_to_idx[label] for label in labels], dtype=np.int64)
        self.tabular_size = tabular_size

    def __getitem__(self, index):
        img_array = self.images[index]

        img_array, tabular = np.split(img_array, [(img_array.shape[-1] - self.tabular_size)])
        img_array = torch.from_numpy(img_array).float()
        tabular = torch.from_numpy(tabular).float()

        assert img_array.shape[-1] == 100
        img_array = img_array.view(-1, 10, 10)
        label = self.labels[index % len(self.labels)]
        return img_array, tabular, label

    def __len__(self):
        return len(self.labels)


class CombinedDataset(Dataset):
    def __init__(self, tabular, labels, cls_to_indx):
        self.tabular = tabular
        self.labels = labels
        self.cls_to_idx = cls_to_indx
        self.labels = np.array([self.cls_to_idx[label] for label in labels], dtype=np.int64)

    def __getitem__(self, index):
        tabular_data = self.tabular[index]
        # TODO: NICER IMPLEMENTATION
        tabular_data_deep, extra_data, log_tabular = np.split(tabular_data, [6, 12])
        tabular_data_shallow = tabular_data_deep.copy()
        log_tabular = torch.from_numpy(log_tabular).float()
        tabular_data_deep = torch.from_numpy(tabular_data_deep).float()
        extra_data = torch.from_numpy(extra_data).float()
        tabular_data_shallow = torch.from_numpy(tabular_data_shallow).float()
        label = self.labels[index % len(self.labels)]
        return (tabular_data_deep, tabular_data_shallow, log_tabular, extra_data), label

    def __len__(self):
        return len(self.labels)


def combine_data(dir_load):
    dir_files = os.listdir(dir_load)
    train_list = []
    valid_list = []
    test_list = []
    column_list = []
    for csv in dir_files:
        if 'train' in csv:
            df = pd.read_csv(os.path.join(dir_load, csv))
            print(f'csv {csv}, len {len(df)}')
            columns = df.columns
            df = df.to_numpy()
            train_list.append(df)
            column_list.append(columns)
        if 'valid' in csv:
            df = pd.read_csv(os.path.join(dir_load, csv))
            print(f'csv {csv}, len {len(df)}')
            df = df.to_numpy()
            valid_list.append(df)
        if 'test' in csv:
            df = pd.read_csv(os.path.join(dir_load, csv))
            print(f'csv {csv}, len {len(df)}')
            df = df.to_numpy()
            test_list.append(df)
    train_data = np.concatenate(train_list, axis=1)
    valid_data = np.concatenate(valid_list, axis=1)
    test_data = np.concatenate(test_list, axis=1)
    column_list = list(np.concatenate(column_list))

    train_data = pd.DataFrame(train_data, columns=column_list)
    train_data = train_data.loc[:, ~train_data.columns.duplicated()]
    train_data = train_data.drop(['Unnamed: 0'], axis=1)
    train_data = train_data.fillna(0)

    valid_data = pd.DataFrame(valid_data, columns=column_list)
    valid_data = valid_data.loc[:, ~valid_data.columns.duplicated()]
    valid_data = valid_data.drop(['Unnamed: 0'], axis=1)
    valid_data = valid_data.fillna(0)

    test_data = pd.DataFrame(test_data, columns=column_list)
    test_data = test_data.loc[:, ~test_data.columns.duplicated()]
    test_data = test_data.drop(['Unnamed: 0'], axis=1)
    test_data = test_data.fillna(0)

    column_list = train_data.columns
    pd.DataFrame(train_data).to_csv("D:\\Projects\\Imbalanced\\Tabular_data\\train_combined.csv", header=column_list,
                                    index=False)
    pd.DataFrame(valid_data).to_csv("D:\\Projects\\Imbalanced\\Tabular_data\\valid_combined.csv", header=column_list,
                                    index=False)
    pd.DataFrame(test_data).to_csv("D:\\Projects\\Imbalanced\\Tabular_data\\test_combined.csv", header=column_list,
                                   index=False)


CLASS_NAME_DICT_5 = {

    11: 'electron',
    13: 'muon',
    211: 'pion',
    321: 'kaon',
    2212: 'proton',
}

CLASS_NAME_DICT_3 = {
    211: 'pion',
    321: 'kaon',
    2212: 'proton',
}
