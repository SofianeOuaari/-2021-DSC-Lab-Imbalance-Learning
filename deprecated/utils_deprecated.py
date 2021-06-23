import random

import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE
from data import TabularDataset
from nn_model import MLP3, MLP1, RANDOM_SEED, CombinedModel
import warnings

# THIS FILE CONTAINS THE DEPRECATED UTILS CODE FOR THE NN MODEL TRAININGS
warnings.warn("This module is deprecated in favour of machine_learning", DeprecationWarning, stacklevel=2)

def init_function(device_in, strategy, ratios, out_dim, model_type, freeze_list,
                  scale=False,
                  ):
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    train_tabular = pd.read_csv(
        "D:\\Projects\\Imbalanced\\Tabular_data\\tabular_train.csv").to_numpy()
    train_data, train_label = train_tabular[:, 1:], (train_tabular[:, 0]).astype(np.int64)

    valid_tabular = pd.read_csv("D:\\Projects\\Imbalanced\\Tabular_data\\tabular_valid.csv").to_numpy()
    valid_data, valid_label = valid_tabular[:, 1:], (valid_tabular[:, 0]).astype(np.int64)

    test_tabular = pd.read_csv("D:\\Projects\\Imbalanced\\Tabular_data\\tabular_test.csv").to_numpy()
    test_data, test_label = test_tabular[:, 1:], (test_tabular[:, 0]).astype(np.int64)

    if model_type == "Combined":
        test_tabular = pd.read_csv("D:\\Projects\\Imbalanced\\Tabular_data\\test_combined.csv").to_numpy()
        test_data, test_label = test_tabular[:, 1:], (test_tabular[:, 0]).astype(np.int64)

        train_tabular = pd.read_csv(
            "D:\\Projects\\Imbalanced\\Tabular_data\\train_combined.csv").to_numpy()
        train_data, train_label = train_tabular[:, 1:], (train_tabular[:, 0]).astype(np.int64)

        valid_tabular = pd.read_csv("D:\\Projects\\Imbalanced\\Tabular_data\\valid_combined.csv").to_numpy()
        valid_data, valid_label = valid_tabular[:, 1:], (valid_tabular[:, 0]).astype(np.int64)

    print(f'Ratio test/all {len(test_data) / (len(test_data) + len(train_data) + len(valid_data))}')
    print(f'Ratio train/all {len(train_data) / (len(test_data) + len(train_data) + len(valid_data))}')
    print(f'Ratio valid/all {len(valid_data) / (len(test_data) + len(train_data) + len(valid_data))}')

    if scale:
        scaling_function = sklearn.preprocessing.StandardScaler()
        scaling_function.fit(train_data)
        train_data = scaling_function.transform(train_data)
        valid_data = scaling_function.transform(valid_data)
        test_data = scaling_function.transform(test_data)
    init_dim = train_data.shape[-1]
    if model_type == 'MLP3':
        model = MLP3(init_dim=init_dim, out_dim=out_dim)
    elif model_type == 'MLP1':
        model = MLP1(init_dim=init_dim, out_dim=out_dim)
    elif model_type == 'Combined':
        model = CombinedModel(6, out_dim=out_dim, freeze_backbone=freeze_list, device=device_in)
    else:
        raise Exception('No such model')

    print(f' Model name {model.__class__.__name__}')
    model.to(device_in)

    split_dict = {'train': select_sampling(train_data, train_label, strategy, ratios),
                  'valid': (valid_data, valid_label),
                  'test': (test_data, test_label)
                  }

    class_stats_f = {phase: {cls: cls_cnt for cls, cls_cnt in class_stats(split_dict[phase][-1]).items()} for phase in
                     split_dict.keys()}

    idx_to_cls = {key_i: key_ for key_i, key_ in enumerate(class_stats_f['train'].keys())}
    cls_to_index = {value_: key_ for key_, value_ in idx_to_cls.items()}
    if model_type == 'Combined':
        dataset_dict = {
            x: CombinedDataset(*split_dict[x], cls_to_indx=cls_to_index) for x in split_dict.keys()}
    else:
        dataset_dict = {
            x: TabularDataset(*split_dict[x], cls_to_indx=cls_to_index) for x in split_dict.keys()}

    data_loaders_func = {phase: DataLoader(dataset_dict[phase], batch_size=BATCH_SIZE, shuffle=True, num_workers=0) for
                         phase in split_dict.keys()}

    return model, class_stats_f, data_loaders_func, dataset_dict, idx_to_cls


if __name__ == '__main__':
    print('Utils main')
