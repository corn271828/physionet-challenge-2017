from pandas import value_counts
import scipy.io as spio
import numpy as np
import random
import os
import shutil

SEED = 65536
random.seed(SEED)

def generate_MCAR(val, proportion=0.1, replace=-9999):
    """ Returns the input with random entries removed.
   
    val        -- 1D integer numpy array
    proportion -- approximate proportion to be removed (default 0.1)
    replace    -- value to replace entries with (default -9999)
    """
    val = val.copy()
    for i in range(val.size):
        if random.random() < proportion:
            val[i] = replace
    return val

def generate_processed_records(processing_function, original_data_directory, new_data_directory):
    """ Generates a processed copy of all the physionet challenge 2017 records.

    processing_function     -- function to process the numpy array values (e.g. generate_MCAR) (function: numpy array -> numpy array)
    original_data_directory -- where to get the validation records (string)
    new_data_directory      -- where to put the new, processed records (string)
    """
    print(f"Attempting to create new directory {new_data_directory} ...")
    try:
        os.mkdir(new_data_directory)
    except FileExistsError:
        print("Oops, directory already exists.")

    print("Copying RECORDS...")
    VALIDATION_RECORDS = os.path.join(original_data_directory, "RECORDS")
    shutil.copy(VALIDATION_RECORDS, new_data_directory)

    print("Generating processed data...")
    with open(VALIDATION_RECORDS) as fs:
        for line in fs:
            line = line.strip()
            
            # Copy over the .hea file
            hea_record = os.path.join(original_data_directory, f"{line}.hea")
            shutil.copy(hea_record, new_data_directory)

            # Load, process, and write records
            cur_record = os.path.join(original_data_directory, line)
            val = spio.loadmat(cur_record, squeeze_me=True)['val']
            val2 = processing_function(val)
            new_record = os.path.join(new_data_directory, f"{line}.mat")
            spio.savemat(new_record, {'val':val2})

    print("Processed data generation complete!")
        
if __name__ == "__main__":
    print(f"Current directory: {os.getcwd()}")
    generate_processed_records(generate_MCAR, "validation", "validation-mcar")

            #print(line, val.size, sum(np.isnan(val)))
            #print(np.unique(val, return_counts=True))
            #print(val.__class__)
            #print(line, val.size, sum(val2 == -9999))