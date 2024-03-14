import pandas as pd
import os

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
        print(c, end='\r')
        c += 1

    for file in file_listB:
        data.append(pd.read_csv(DATA_PATH + 'training_setB/' + file))
        print(c, end='\r')
        c += 1
    
    print("Putting data into dataframe...")
    dataset = pd.concat(data)
    print("Done")

    return dataset
