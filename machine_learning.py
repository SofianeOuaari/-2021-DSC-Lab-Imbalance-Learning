import os.path
import time

import numpy as np
import pandas as pd
import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utils import class_stats, print_class_rep


def grid_search_model(model, train_x, train_y, valid_x, valid_y):
    """
    Grid search for model parameters
    :param model: given model (Random Forest or Gradient Boosting)
    :param train_x: train data
    :param train_y: train labels
    :param valid_x: valid data
    :param valid_y: valid labels
    """
    grid = {
        'n_estimators': [100, 175, 250],
        'min_samples_split': [2, 5],
        'max_depth': [1, 3, 5],
    }
    best_minority_f1 = 0.0
    best_grid = None
    for g in tqdm.tqdm(ParameterGrid(grid), 'Iterating parameters'):
        print(g)
        model.set_params(**g)
        model.fit(train_x, train_y)

        # eval model
        predictions = model.predict(valid_x)
        class_rep = classification_report(y_true=valid_y, y_pred=predictions, output_dict=True, zero_division=0)

        if class_rep['0']['f1-score'] > best_minority_f1:
            best_minority_f1 = class_rep['0']['f1-score']
            best_grid = g

    print(best_minority_f1, best_grid)


def train_val(model, train_x, train_y, valid_x, valid_y, idx_to_cls):
    """
    Simple train and evaluation function
    :param model: given model (Random Forest or Gradient Boosting)
    :param train_x: train data
    :param train_y: train labels
    :param valid_x: valid data
    :param valid_y: valid labels
    :param idx_to_cls: label encoding
    """
    print('Running train val')
    model.fit(train_x, train_y)
    # eval model
    predictions = model.predict(valid_x)
    class_rep = classification_report(y_true=valid_y, y_pred=predictions, output_dict=True, zero_division=0)

    print_class_rep(class_rep, 'f1-score', idx_to_cls)
    print(class_rep)


def ovo_model(model, train_x, train_y, valid_x, valid_y, idx_to_cls):
    """
    One vs One model training end evaluation
    :param model: given model (Random Forest or Gradient Boosting)
    :param train_x: train data
    :param train_y: train labels
    :param valid_x: valid data
    :param valid_y: valid labels
    :param idx_to_cls: label encoding
    """
    print('Running One vs One')
    start_time = time.time()
    clf = OneVsOneClassifier(model).fit(train_x, train_y)
    time_diff = (time.time() - start_time)
    print(f'Time = {time_diff}!')
    predictions = clf.predict(valid_x)
    class_rep = classification_report(y_true=valid_y, y_pred=predictions, output_dict=True, zero_division=0)

    print_class_rep(class_rep, 'f1-score', idx_to_cls)
    print(class_rep)


def ova_model(model, train_x, train_y, valid_x, valid_y, idx_to_cls):
    """
    One vs All model training end evaluation
    :param model: given model (Random Forest or Gradient Boosting)
    :param train_x: train data
    :param train_y: train labels
    :param valid_x: valid data
    :param valid_y: valid labels
    :param idx_to_cls: label encoding
    """
    print('Running One vs All')
    start_time = time.time()
    clf = OneVsRestClassifier(model).fit(train_x, train_y)
    time_diff = (time.time() - start_time)
    print(f'Time = {time_diff}!')
    predictions = clf.predict(valid_x)
    class_rep = classification_report(y_true=valid_y, y_pred=predictions, output_dict=True, zero_division=0)

    print_class_rep(class_rep, 'f1-score', idx_to_cls)
    print(class_rep)


def bagging_model(model, train_x, train_y, valid_x, valid_y):
    """
    Bagging model training end evaluation
    :param model: given model (Random Forest or Gradient Boosting)
    :param train_x: train data
    :param train_y: train labels
    :param valid_x: valid data
    :param valid_y: valid labels
    """
    n_estimators = 20
    clf = OneVsOneClassifier(BaggingClassifier(model, max_samples=1.0 / n_estimators,
                                               n_estimators=n_estimators, n_jobs=10))
    clf.fit(train_x, train_y)
    predictions = clf.predict(valid_x)
    class_rep = classification_report(y_true=valid_y, y_pred=predictions, output_dict=True, zero_division=0)

    class_stats_f = class_stats(valid_y)
    idx_to_cls = {key_i: key_ for key_i, key_ in enumerate(class_stats_f.keys())}

    print_class_rep(class_rep, 'f1-score', idx_to_cls)


def baseline_train(path_train, path_valid, ):
    """
    Baseline for multiple models
    :param path_train: path to train data
    :param path_valid: path to valid data
    """
    train_tabular = pd.read_csv(path_train)

    train_tabular = train_tabular.to_numpy()
    train_data, train_label = train_tabular[:, 1:], (train_tabular[:, 0]).astype(np.int64)

    classifier_list = [
        XGBClassifier(use_label_encoder=False, n_jobs=14),
        DecisionTreeClassifier(),
        RandomForestClassifier(verbose=2, n_jobs=14, ),
        AdaBoostClassifier(),
        GradientBoostingClassifier(verbose=2)
    ]

    # class renaming
    class_stats_f = class_stats(train_label)
    print(class_stats_f)

    # label encoding
    # TODO CHANGE TO SKLEARN ENCODER
    idx_to_cls = {key_i: key_ for key_i, key_ in enumerate(class_stats_f.keys())}
    cls_to_index = {value_: key_ for key_, value_ in idx_to_cls.items()}
    train_label = np.array([cls_to_index[label] for label in train_label], dtype=np.int64)

    valid_tabular = pd.read_csv(path_valid).to_numpy()
    valid_data, valid_label = valid_tabular[:, 1:], (valid_tabular[:, 0]).astype(np.int64)
    valid_label = np.array([cls_to_index[label] for label in valid_label], dtype=np.int64)

    scale = True
    # scaling
    if scale:
        scaling_function = StandardScaler()
        scaling_function.fit(train_data)
        train_data = scaling_function.transform(train_data)
        valid_data = scaling_function.transform(valid_data)

    final_results_dict = {}
    for model in tqdm.tqdm(classifier_list, f"Iterating models"):
        start_time = time.time()
        model.fit(train_data, train_label)
        time_diff = (time.time() - start_time)
        predictions_valid = model.predict(valid_data)

        class_rep = classification_report(y_true=valid_label, y_pred=predictions_valid, output_dict=True,
                                          zero_division=0)

        # Dict creation
        final_results_dict[f'{model.__class__.__name__}'] = [class_rep['macro avg']['f1-score'],
                                                             class_rep['0']['f1-score'], class_rep['1']['f1-score'],
                                                             class_rep['2']['f1-score'], class_rep['3']['f1-score'],
                                                             time_diff, time_diff / 60.0]

    cols = ['valid_macro_f1', '-11', '211', '321', '2212', 'train_time_sec', 'train_time_min']
    # saving results
    final_df = pd.DataFrame.from_dict(final_results_dict, orient='index', columns=cols)
    final_df.to_csv(f"D:\\University\\ELTE\\2.Semester\\DSC_Lab\\train_set_results.csv")


def run_function(path_train, path_valid, function):
    """
    Running the given functions like ovo and ova and train_eval, data loader
    :param path_train: train data path
    :param path_valid: valid data path
    :param function: function to execute
    """
    train_tabular = pd.read_csv(path_train)

    train_tabular = train_tabular.to_numpy()
    train_data, train_label = train_tabular[:, 1:], (train_tabular[:, 0]).astype(np.int64)
    class_stats_f = class_stats(train_label)
    print(class_stats_f)

    idx_to_cls = {key_i: key_ for key_i, key_ in enumerate(class_stats_f.keys())}
    cls_to_index = {value_: key_ for key_, value_ in idx_to_cls.items()}
    train_label = np.array([cls_to_index[label] for label in train_label], dtype=np.int64)

    valid_tabular = pd.read_csv(path_valid).to_numpy()
    valid_data, valid_label = valid_tabular[:, 1:], (valid_tabular[:, 0]).astype(np.int64)
    valid_label = np.array([cls_to_index[label] for label in valid_label], dtype=np.int64)

    scale = True
    if scale:
        scaling_function = StandardScaler()
        scaling_function.fit(train_data)
        train_data = scaling_function.transform(train_data)
        valid_data = scaling_function.transform(valid_data)

    # Grid search result
    params = {'max_depth': 25, 'min_samples_split': 10, 'n_estimators': 300, 'n_jobs': 14}
    model = RandomForestClassifier(verbose=2, **params)
    function(model, train_data, train_label, valid_data, valid_label, idx_to_cls)


if __name__ == '__main__':
    run_function("tabular_train.csv",
                 "tabular_test.csv", train_val)
