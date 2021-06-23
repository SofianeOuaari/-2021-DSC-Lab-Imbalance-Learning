import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler

from utils import class_stats, print_class_rep

train_tabular = pd.read_csv("D:\\Projects\\Imbalanced\\Tabular_data\\tabular_train.csv")
column_names = train_tabular.columns
# Loading in the tabular train data csv
train_tabular = train_tabular.to_numpy()
train_data, train_label = train_tabular[:, 1:], (train_tabular[:, 0]).astype(np.int64)
# Loading in the tabular valid data csv
valid_tabular = pd.read_csv("D:\\Projects\\Imbalanced\\Tabular_data\\tabular_valid.csv").to_numpy()
valid_data, valid_label = valid_tabular[:, 1:], (valid_tabular[:, 0]).astype(np.int64)

# class renaming
# TODO: CHANGE TO LABEL ENCODER SKLEARN
class_stats_f = class_stats(train_label)
print(f'Smote stats {class_stats_f}')
idx_to_cls = {key_i: key_ for key_i, key_ in enumerate(class_stats_f.keys())}
cls_to_index = {value_: key_ for key_, value_ in idx_to_cls.items()}

train_label = np.array([cls_to_index[label] for label in train_label], dtype=np.int64)
valid_label = np.array([cls_to_index[label] for label in valid_label], dtype=np.int64)

# query 750 new elements each iteration
query_new = 750

# randomly selecting 500 element from each class
new_train = np.empty((0, 6))
new_label = np.empty(0)
sample_starting = 500
for key in idx_to_cls.keys():
    temp_x = train_data[train_label == key][:sample_starting]
    new_train = np.concatenate((new_train, temp_x), axis=0)
    new_label = np.concatenate((new_label, (np.ones(sample_starting) * key)), axis=0)
# random shuffle
perm = torch.randperm(len(new_label))
new_train = new_train[perm]
new_label = new_label[perm].astype(np.int64)
print(f'Original stats: {class_stats(train_label)}')

# standard scaling all the datasets with the transformation learned on train
scale = False
if scale:
    scaling_function = StandardScaler()
    scaling_function.fit(train_data)
    train_data = scaling_function.transform(train_data)
    new_train = scaling_function.transform(new_train)
    valid_data = scaling_function.transform(valid_data)

# initializing the learner, already fits it to the initial data
learner = ActiveLearner(
    estimator=RandomForestClassifier(n_jobs=14),
    query_strategy=uncertainty_sampling,
    X_training=new_train, y_training=new_label
)
# number of iterations and logging the best
epochs = 150
best_minority_f1 = 0.0
plot = False
best_labels = None
best_train = None

start_time = time.time()
for epoch in range(epochs):

    # fitting learner
    model = learner.estimator
    predictions = model.predict(valid_data)
    # results
    class_rep = classification_report(y_true=valid_label, y_pred=predictions, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(valid_label, predictions)

    if plot:
        display = plot_confusion_matrix(model, valid_data, valid_label,
                                        display_labels=['positron', 'pion', 'kaon', 'proton'])
        plt.show()

    print(f'EPOCH {epoch}')
    print_class_rep(class_rep, 'f1-score', idx_to_cls)
    print(f'Current best minority f1: {best_minority_f1}')

    if class_rep['0']['f1-score'] > best_minority_f1:
        best_labels = learner.y_training
        best_train = learner.X_training
        best_minority_f1 = class_rep['0']['f1-score']

    # query for new labels and instances from the train data
    query_idx, query_inst = learner.query(train_data, n_instances=query_new)

    # ...obtaining new labels from the Oracle... supply label for queried instance
    learner.teach(train_data[query_idx], train_label[query_idx])
    print(f'New stats: {class_stats(learner.y_training)}')

time_diff = (time.time() - start_time)
print(f'Time elapsed sec: {time_diff} min: {time_diff / 60}')
y_label = np.array([idx_to_cls[label] for label in best_labels], dtype=np.int64)
train_array = np.concatenate((np.expand_dims(y_label, axis=1), best_train), axis=1)
# saving best active learning undersampled dataset
pd.DataFrame(train_array).to_csv("AL_sampling.csv", header=column_names, index=False)
