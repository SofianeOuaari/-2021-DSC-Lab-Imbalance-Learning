import random

import numpy as np
import torch

from data import CLASS_NAME_DICT_4

RANDOM_SEED = 32


def select_sampling(x_train, y_train, strategy, ratios):
    """
    Select the correct sampling method
    :param x_train: train data
    :param y_train: train labels
    :param strategy: sampling strategy (over, under, original)
    :param ratios: sampling ratio
    :return:
    """
    if ratios > 0 and strategy == 'under':
        x_train, y_train = under_sampling(data=x_train, targets=y_train, ratio=ratios)
    elif ratios > 0 and strategy == 'over':
        x_train, y_train = over_sampling(data=x_train, targets=y_train, ratio=ratios)
    elif strategy == 'original':
        return x_train, y_train
    else:
        raise Exception('Not correct sampling method')
    return x_train, y_train


def random_under_sample(data, labels_in, y_class_list, majority_key, n=100):
    """
    Selects n random datapoints and their corresponding labels from a dataset, under sampling
    :param data: train data
    :param labels_in: train labels
    :param y_class_list: classes to sample
    :param majority_key: majority key to under sample
    :param n: number of data points to keep
    :return:
    """
    random.seed(RANDOM_SEED)
    final_labels = np.empty((0,)).astype(np.int64)

    if len(data.shape) >= 2:
        final_data = np.empty((0, data.shape[-1]))
    else:
        final_data = np.empty((0,))
    assert len(data) == len(labels_in)
    for class_value in y_class_list:
        new_data = data[labels_in == class_value]
        new_labels = labels_in[labels_in == class_value]
        # TODO: CURRENTLY IT UNDER SAMPLES BOTH OTHER
        if class_value in majority_key:
            perm = torch.randperm(len(new_data))
            new_data, new_labels = new_data[perm][:n], new_labels[perm][:n]

        final_data = np.concatenate((final_data, new_data))
        final_labels = np.concatenate((final_labels, new_labels))

    return final_data, final_labels


def random_over_sample(data, labels_in, y_class_list, missing_dict):
    """
    Selects n random datapoints and their corresponding labels from a dataset, oversampling
    :param data: train data
    :param labels_in: train labels
    :param y_class_list: classes to sample
    :param missing_dict: missing elements to oversample
    :return:
    """

    index_list_final = []
    assert len(data) == len(labels_in)
    for class_value in y_class_list:
        indexes = np.where(labels_in == class_value)[0]
        random_list = random.choices(indexes, k=missing_dict[class_value])
        index_list_final += random_list
    final_data = np.concatenate((data[index_list_final], data))
    final_labels = np.concatenate((labels_in[index_list_final], labels_in))
    return final_data, final_labels


def under_sampling(data, targets, ratio):
    """
    Encapsulating function for undersampling
    :param data: data
    :param targets: labels
    :param ratio: undersampling ratios
    :return:
    """
    statistics_dict = class_stats(targets)
    sorted_vals = sorted(statistics_dict.values())
    minority_count = int(sorted_vals[1] * ratio)
    majority_key = [211, 2212]
    final_data, final_label = random_under_sample(data, targets, statistics_dict.keys(), majority_key, minority_count)
    return final_data, final_label


def over_sampling(data, targets, ratio):
    """
    Encapsulating function for over sampling
    :param data: data
    :param targets: labels
    :param ratio: over sampling ratios
    :return:
    """
    statistics_dict = class_stats(targets)
    majority_key = max(statistics_dict, key=statistics_dict.get)
    majority_count = int(statistics_dict[majority_key] * ratio)
    missing_dict = {key_: abs(majority_count - statistics_dict[key_]) for key_ in
                    statistics_dict.keys() - [majority_key]}
    final_data, final_label = random_over_sample(data, targets, statistics_dict.keys() - [majority_key], missing_dict)
    return final_data, final_label


def class_stats(y, change_str=False):
    """
    Class count
    :param y: labels
    :param change_str: original label names
    :return:
    """
    keys, values = np.unique(y, return_counts=True)
    dict_stat = {key_: value_ for key_, value_ in zip(keys, values)}
    if change_str:
        dict_stat = {CLASS_NAME_DICT_4[key_]: value_ for key_, value_ in zip(keys, values)}
    return dict_stat


def print_class_rep(class_report, eval_crit, idx_to_cls_dict):
    """
    Print function
    :param class_report: eval report
    :param eval_crit: criterion
    :param idx_to_cls_dict: index to label encoding
    """
    additional_keys = ['precision', 'recall']
    for key_c in idx_to_cls_dict.keys():
        print(f'class {idx_to_cls_dict[key_c]} {eval_crit}: {class_report[str(key_c)][eval_crit]}, '
              f'precision {class_report[str(key_c)][additional_keys[0]]}, '
              f'recall {class_report[str(key_c)][additional_keys[1]]}')


if __name__ == '__main__':
    print('Utils main')
