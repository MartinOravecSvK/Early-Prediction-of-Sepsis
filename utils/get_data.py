import pandas as pd
import numpy as np
import hypertools as hyp
import os
from copy import copy

import warnings
warnings.filterwarnings("ignore")

# Returns the absolute path of the dataset directory.
def get_dataset_abspath():
    abs_script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(abs_script_path)
    parent_dir = os.path.dirname(script_dir)
    DATA_PATH = parent_dir + '/Dataset/'

    return DATA_PATH

def get_dataset_as_df():
    data = []
    c = 0
    DATA_PATH = get_dataset_abspath()
    file_listA = os.listdir(DATA_PATH + 'training_setA/')
    file_listB = os.listdir(DATA_PATH + 'training_setB/')

    for file in file_listA:
        data.append(pd.read_csv(DATA_PATH + 'training_setA/' + file))
        print("  ", c, end='\r')
        c += 1

    for file in file_listB:
        data.append(pd.read_csv(DATA_PATH + 'training_setB/' + file))
        print("  ", c, end='\r')
        c += 1
    
    print("Putting data into dataframe...")
    # dataset = pd.concat(data)
    dataset = []
    print("Done")

    return dataset

def get_dataset_as_np():
    data = []
    c = 0
    DATA_PATH = get_dataset_abspath()  # Make sure this function is defined somewhere in your code
    file_listA = os.listdir(DATA_PATH + 'training_setA/')
    file_listB = os.listdir(DATA_PATH + 'training_setB/')

    for file in file_listA:
        data.append(pd.read_csv(DATA_PATH + 'training_setA/' + file))
        print("  ", c, end='\r')
        c += 1

    for file in file_listB:
        data.append(pd.read_csv(DATA_PATH + 'training_setB/' + file))
        print("  ", c, end='\r')
        c += 1

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(data, ignore_index=True)
    
    print("Putting data into numpy array...")
    # Convert the combined DataFrame into a numpy array
    dataset = combined_df.to_numpy()
    print("Done")

    return dataset

if __name__ == '__main__':
    data1 = get_dataset_as_np()
    data2 = copy(data1)

    missing = .1
    inds = [(i,j) for i in range(data2.shape[0]) for j in range(data2.shape[1])]
    missing_data = [inds[i] for i in np.random.choice(int(len(inds)), int(len(inds)*missing))]
    for i,j in missing_data:
        data2[i,j]=np.nan

    # plot
    hyp.plot([data1, data2], linestyle=['-',':'], legend=['Original', 'PPCA'])