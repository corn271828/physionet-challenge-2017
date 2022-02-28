# the code here is based off of Tianqi Zhou's code at https://github.com/tianqi-zhou/mimic_imputation.
from multiprocessing.spawn import prepare
import numpy as np
from imputator import generate_MCAR
import random

def _generate_train_val_split(N, seed, trainval_ratio=0.8):
    """ Generates two sets of IDs train_ids and val_ids, where IDs are 
    indices from 0 to N

    N              -- number of data points
    seed           -- random seed
    trainval_ratio -- the proportion of IDs that are in train_ids
    """

    # Split train/val dat
    np.random.seed(seed) # set random seed
    permuted_ids = np.random.permutation(N)
    train_ids, val_ids = permuted_ids[:int(N * trainval_ratio)], permuted_ids[int(N * trainval_ratio):]

    return train_ids, val_ids

def prepare_imputation_data(data, seed=1001):
    """ Prepares training and validation data.

    data -- list of numpy arrays
    seed -- random seed
    """
    series_data = data
    series_label = np.copy(series_data)

    # Create missing data
    random.seed(seed)
    series_data = [generate_MCAR(series) for series in series_data]

    # Generate train/val splits
    train_ids, val_ids = _generate_train_val_split(len(data), seed)

    train_data = [series_data[id] for id in train_ids]
    train_label = [series_label[id] for id in train_ids]

    val_data = [series_data[id] for id in val_ids]
    val_label = [series_label[id] for id in val_ids]

    return train_data, train_label, val_data, val_label

if __name__ == "__main__":
    data = [np.random.uniform(size=10) for _ in range(20)]
    train_data, train_label, val_data, val_label = prepare_imputation_data(data)
    print(train_data, train_label, val_data, val_label)
