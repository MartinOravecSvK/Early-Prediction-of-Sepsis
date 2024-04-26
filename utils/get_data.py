import pandas as pd
import numpy as np
import hypertools as hyp
import os
import matplotlib.pyplot as plt
from ppca import PPCA

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
        data.append(pd.read_csv(DATA_PATH + 'training_setA/' + file, sep='|'))
        print("  ", c, end='\r')
        c += 1
    print("  ", c)
    for file in file_listB:
        data.append(pd.read_csv(DATA_PATH + 'training_setB/' + file, sep='|'))
        print("  ", c, end='\r')
        c += 1
    print("  ", c)
    
    print("Putting data into dataframe...")
    dataset = pd.concat(data)
    print("Done")

    return dataset

def get_dataset_as_np(include_strings=False):
    data = []
    c = 0
    DATA_PATH = get_dataset_abspath()  # Ensure this function is defined and returns the correct path
    file_listA = os.listdir(DATA_PATH + 'training_setA/')
    file_listB = os.listdir(DATA_PATH + 'training_setB/')

    for file in file_listA:
        df = pd.read_csv(DATA_PATH + 'training_setA/' + file, na_values=['NA', 'na', ''])
        data.append(df)
        print("  ", c, end='\r')
        c += 1

    for file in file_listB:
        df = pd.read_csv(DATA_PATH + 'training_setB/' + file, na_values=['NA', 'na', ''])
        data.append(df)
        print("  ", c, end='\r')
        c += 1
    print("  ", c)

    print("Combining data into a single DataFrame and converting to numpy array...")
    # print(data.shape)
    combined_df = pd.concat(data)
    print(combined_df.shape)
    # Ensure the data is numeric and convert to float
    numeric_df = combined_df.apply(pd.to_numeric, errors='coerce')
    dataset = numeric_df.astype(float).to_numpy()

    if not include_strings:
        dataset = preprocess_no_strings(combined_df)

    print("Done")
    return dataset

def get_dataset():
    data = []
    patient_ids = np.array([])
    DATA_PATH = get_dataset_abspath()
    file_listA = os.listdir(DATA_PATH + 'training_setA/')
    file_listB = os.listdir(DATA_PATH + 'training_setB/')
    patient_counter = 1  # Start counter for patient IDs
    patient_id_map = {}  # Map to store counter to filename mapping
    
    # Process files in training_setA
    for file_name in file_listA:
        file_path = os.path.join(DATA_PATH, 'training_setA', file_name)
        df_temp = pd.read_csv(file_path, sep='|')
        patient_ids = np.append(patient_ids, [patient_counter] * len(df_temp))  # Use counter as patient ID
        # patient_ids.extends([patient_counter] * len(df_temp))  # Use counter as patient ID
        data.append(df_temp)
        patient_id_map[patient_counter] = file_name  # Map counter to filename
        patient_counter += 1  # Increment counter for the next patient
        print("  ", patient_counter, end='\r')
    print("  ", patient_counter)

    # Process files in training_setB
    for file_name in file_listB:
        file_path = os.path.join(DATA_PATH, 'training_setB', file_name)
        df_temp = pd.read_csv(file_path, sep='|')
        patient_ids = np.append(patient_ids, [patient_counter] * len(df_temp))
        # patient_ids.extend([patient_counter] * len(df_temp))  # Use counter as patient ID
        data.append(df_temp)
        patient_id_map[patient_counter] = file_name  # Map counter to filename
        patient_counter += 1  # Increment counter for the next patient
        print("  ", patient_counter, end='\r')
    print("  ", patient_counter)

    combined_df = pd.concat(data, ignore_index=True)
    combined_df['patient_id'] = patient_ids
    combined_df.set_index(['patient_id', combined_df.index], inplace=True)
    
    print("Dataset loaded into a MultiIndex DataFrame.")
    return combined_df, patient_id_map

def preprocess_no_strings(df):
    """
    Preprocess the given DataFrame by removing non-numeric columns and 
    converting the result to a NumPy array of floats.
    
    Parameters:
    - df: pandas DataFrame containing the dataset to preprocess.

    Returns:
    - A NumPy array containing only the numeric data from the input DataFrame.
    """
    print("Filtering only numeric data columns...")
    # Select columns with dtype 'number', which includes int and float but excludes object/string types
    # print(df.shape)
    numeric_df = df.select_dtypes(include=['number'])
    # print(numeric_df.shape)
    print("Converting to numpy array...")
    numeric_array = numeric_df.astype(float).to_numpy()

    print("Preprocessing complete.")
    return numeric_array

def plot_loadings(loadings, feature_names, pc1=0, pc2=1):
    """
    Plot the loadings for the principal components.

    Parameters:
    - loadings: The loadings matrix from PPCA.
    - feature_names: A list of names corresponding to the features.
    - pc1: Index of the first principal component to plot.
    - pc2: Index of the second principal component to plot.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot each loading vector. You can also annotate it with the feature name.
    for i, feature in enumerate(feature_names):
        ax.arrow(0, 0, loadings[i, pc1], loadings[i, pc2], head_width=0.02, head_length=0.03, fc='r', ec='r')
        ax.text(loadings[i, pc1]* 1.15, loadings[i, pc2] * 1.15, feature, color='green', ha='center', va='center')

    plt.xlabel(f'PC{pc1+1}')
    plt.ylabel(f'PC{pc2+1}')
    plt.title('Feature contributions to the Principal Components')
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()

if __name__ == '__main__':
    print("Testing get_dataset_as_df...")