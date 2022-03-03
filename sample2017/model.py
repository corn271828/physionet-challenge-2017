# the code here is based off of Tianqi Zhou's code at https://github.com/tianqi-zhou/mimic_imputation.
import numpy as np
from imputator import generate_MCAR, get_some_data
import random
from preprocessor import Preprocessor

def _generate_train_val_split(N, seed, trainval_ratio=0.8):
    """ Generates two sets of IDs train_ids and val_ids, where IDs are 
    indices from 0 to N

    N              -- number of data points
    seed           -- random seed
    trainval_ratio -- the proportion of total IDs that are in train_ids
    """

    # Split train/val dat
    np.random.seed(seed) # set random seed
    permuted_ids = np.random.permutation(N)
    train_ids, val_ids = permuted_ids[:int(N * trainval_ratio)], permuted_ids[int(N * trainval_ratio):]

    return train_ids, val_ids

def rectangularize(data, pad=np.nan):
    """ Turns a list of numpy arrays with different lengths into a two-d numpy array, by
    padding the right side of shorter arrays with 'pad'. Modifies the original input.

    data -- list of numpy arrays
    pad  -- value to pad with
    """

    maxLength = 0
    for array in data:
        if (array.size) > maxLength:
            maxLength = array.size
    data = [np.pad(array.astype(float), (0, maxLength - array.size), constant_values=pad) for array in data] # sadly, need to convert to float to insert nans

    return np.stack(data, axis=0)

def prepare_imputation_data(data, seed=1001):
    """ Prepares training and validation data.

    data -- list of numpy arrays (arrays may have different lengths)
    seed -- random seed
    """
    cuts = [array.size for array in data] # original lengths before padding.
    data = rectangularize(data) 
    # data is now a 2d numpy array. this is much faster than a list of arrays.

    series_data = data
    series_label = np.copy(series_data)

    # Create missing data
    random.seed(seed)
    series_data = np.apply_along_axis(lambda x: generate_MCAR(x, 0.1, np.nan), axis=1, arr=series_data)

    # Generate train/val splits
    train_ids, val_ids = _generate_train_val_split(len(data), seed)

    train_data = series_data[train_ids]
    train_label = series_label[train_ids]

    val_data = series_data[val_ids]
    val_label = series_label[val_ids]

    return train_data, train_label, val_data, val_label, cuts

if __name__ == "__main__":
    #data = [np.random.uniform(size=size) for size in range(10, 20)]
    data = get_some_data("training2017", 10)
    print(data)

    train_data, train_label, val_data, val_label, cuts = prepare_imputation_data(data)
    print(train_data, train_label, val_data, val_label, cuts)

    processor = Preprocessor('normal')
    processor.fit(train_data)
    train_data_scaled = processor.preprocess(train_data)
    train_label_scaled  = processor.preprocess(train_label)
    val_label_scaled  = processor.preprocess(val_label)

    print(train_data_scaled)

