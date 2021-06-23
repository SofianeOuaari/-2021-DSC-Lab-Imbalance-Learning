import numpy as np
import warnings

# THIS FILE CONTAINS THE DEPRECATED CODE FOR THE ACTIVE LEARNING PART
warnings.warn("This module is deprecated in favour of active_learning", DeprecationWarning, stacklevel=2)

def mislabeled_uncertainty(classifier, x_data, y_data):
    # calculate uncertainty for each point provided
    class_wise_uncertainty = classifier.predict_proba(x_data)
    class_wise_predictions = np.argmax(class_wise_uncertainty, axis=1)
    correct_preds = np.where(class_wise_predictions == y_data)[0]
    # for each point, select the maximum uncertainty
    uncertainty = np.max(class_wise_uncertainty, axis=1)
    uncertainty[correct_preds] = uncertainty[correct_preds] * 0
    return uncertainty


def new_query(classifier, x_data, y_data, n_instances: int = 1):
    uncertainty = mislabeled_uncertainty(classifier, x_data, y_data)
    max_idx = np.argpartition(-uncertainty, n_instances - 1, axis=0)[:n_instances]
    # remove it later
    max_idx = uncertainty[uncertainty != 0]
    return max_idx

