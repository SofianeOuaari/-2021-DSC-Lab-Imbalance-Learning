from __future__ import print_function, division

import copy
import os
import random
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import STEP, baseline, EPOCH
from loss import FocalLoss, LabelSmoothingCrossEntropy
from nn_model import RANDOM_SEED
from utils import init_function
import warnings

# THIS FILE CONTAINS THE DEPRECATED CODE FOR TRAINING THE MODELS
warnings.warn("This module is deprecated in favour of machine_learning", DeprecationWarning, stacklevel=2)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer_ft, scheduler, num_epochs, data_loaders, summary_writer,
                dataset_list_input, idx_to_class, combined):
    """
    Deprecated train model function
    :param model: model as input object
    :param criterion: loss function object
    :param optimizer_ft: optimizer object
    :param scheduler: scheduler object
    :param num_epochs: number of iterations to run
    :param data_loaders: pytorch data loaders
    :param summary_writer: tensorboard logger
    :param dataset_list_input: pytorch dataset objects
    :param idx_to_class: class to id encoding dict
    :param combined: parameter for combined backbone model
    :return:
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_macro_f1 = 0.0
    class_names = [str(value) for value in idx_to_class.values()]
    all_labels = [key for key in idx_to_class.keys()]

    checkpoint_name = '_'.join(summary_writer.get_logdir().split('\\')[-2:])

    for epoch_i in tqdm(range(num_epochs), f'Iterating epochs of run'):

        for phase in dataset_list_input:

            # Set model to training mode
            if phase in ['valid', 'test']:
                model.eval()  # Set model to evaluate mode
            if phase == 'train':
                model.train()

            predictions = []
            ground_truth = []
            running_loss = 0.0
            # Iterate over data
            for batch_i, (tabular, labels) in enumerate(data_loaders[phase]):
                labels = labels.to(DEVICE)
                if combined:
                    tabular_data_deep, tabular_data_shallow, log_tabular, extra_data = tabular
                    tabular_data_deep = tabular_data_deep.to(DEVICE)
                    tabular_data_shallow = tabular_data_shallow.to(DEVICE)
                    log_tabular = log_tabular.to(DEVICE)
                    extra_data = extra_data.to(DEVICE)
                else:
                    tabular = tabular.to(DEVICE)

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if combined:
                        outputs = model(tabular_data_deep, tabular_data_shallow, log_tabular, extra_data)
                    else:
                        outputs = model(tabular)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()
                if combined:
                    running_loss += loss.item() * tabular[0].size(0)
                else:
                    running_loss += loss.item() * tabular.size(0)
                predictions.append(preds.detach().cpu().numpy())
                ground_truth.append(labels.detach().cpu().numpy())

            finale_prediction = np.concatenate(predictions, axis=0)
            finale_ground_truth = np.concatenate(ground_truth, axis=0)

            epoch_loss = running_loss / len(finale_ground_truth)

            class_rep = classification_report(y_true=finale_ground_truth, y_pred=finale_prediction, labels=all_labels,
                                              target_names=class_names, output_dict=True, zero_division=0)
            if phase == 'train':
                scheduler.step()

            # # deep copy the model
            if (phase == 'valid') and (class_rep['macro avg']['f1-score'] > best_macro_f1):
                best_macro_f1 = class_rep['macro avg']['f1-score']
                print(f'Current best {best_macro_f1}')
                best_model_wts = copy.deepcopy(model.state_dict())

            summary_writer.add_scalar(f"loss/{phase}_loss", epoch_loss, epoch_i)
            summary_writer.add_scalar(f"metrics/{phase}_macro_f1", class_rep['macro avg']['f1-score'], epoch_i)
            if len(idx_to_class) > 2:
                summary_writer.add_scalar(f"metrics/{phase}_micro_f1", class_rep['accuracy'], epoch_i)
            class_names = [str(value) for value in idx_to_class.values()]
            for k, v in class_rep.items():
                if k in class_names:
                    summary_writer.add_scalar(f"metrics_class_{phase}/{int(k)}_f1", v['f1-score'], epoch_i)

    summary_writer.close()

    # SAVE BEST MODEL
    torch.save(best_model_wts, os.path.join(os.getcwd(), 'data', 'checkpoints', f'{checkpoint_name}.pt'))
    return model


def full_train_init():
    """
    Deprecated train initialization function
    """
    # FIX RANDOM SEEDS
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    dataset_list = ['train', 'valid', 'test']

    param_values = [v for v in baseline.values()]
    epsilon = 0.3

    for params_i, (lr, decay, optimizer, loss_crit, rate, m_type, strat) in enumerate(
            list(product(*param_values))):

        model_g, c_stats, data_load, data_sets, idx_to_cls = init_function(DEVICE,
                                                                           model_type=m_type,
                                                                           strategy=strat,
                                                                           ratios=rate,
                                                                           out_dim=4,
                                                                           scale=True,
                                                                           freeze_list=[])
        print(c_stats)
        print(f'Loss type {loss_crit}')
        writer = SummaryWriter(
            os.path.join(os.getcwd(), 'data', 'runs', 'tabular', '4class', 'baseline', f'{strat}_{rate}_m_{m_type}_workers14'))

        if optimizer == 'SGD':
            optimizer_method = optim.SGD(model_g.parameters(), lr=lr)
        elif optimizer == 'Adam':
            optimizer_method = optim.Adam(model_g.parameters(), lr=lr)
        else:
            raise Exception(f'No optimizer such as {optimizer}')

        if loss_crit == 'CE':
            loss_crit = nn.CrossEntropyLoss()
        elif loss_crit == 'Focal':
            loss_crit = FocalLoss()
        elif loss_crit == 'smooth':
            loss_crit = LabelSmoothingCrossEntropy(epsilon=epsilon)
        else:
            raise Exception(f'No loss such as {loss_crit}')

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_method, step_size=STEP, gamma=decay)

        train_model(model_g, criterion=loss_crit, optimizer_ft=optimizer_method, scheduler=exp_lr_scheduler,
                    num_epochs=EPOCH, data_loaders=data_load,
                    summary_writer=writer, dataset_list_input=dataset_list,
                    idx_to_class=idx_to_cls, combined=False)


if __name__ == '__main__':

    full_train_init()
