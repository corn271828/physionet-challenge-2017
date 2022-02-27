from pandas import value_counts
import scipy.io as spio
import numpy as np

CURRENT_DIRECTORY = '/Users/piercewlai/Documents/stuff/urop/physionet-2017-challenge/sample2017/'
VALIDATION_RECORDS = CURRENT_DIRECTORY + 'validation/RECORDS'

with open(VALIDATION_RECORDS) as fs:
    for line in fs:
        line = line.strip()
        val = spio.loadmat(CURRENT_DIRECTORY + 'validation/' + line, squeeze_me=True)['val']
        print(line, val.size, sum(np.isnan(val)))
        #print(np.unique(val, return_counts=True))

