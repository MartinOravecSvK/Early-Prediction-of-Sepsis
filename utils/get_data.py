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
        data.append(pd.read_csv(DATA_PATH + 'training_setA/' + file))
        print("  ", c, end='\r')
        c += 1
    print("  ", c)
    for file in file_listB:
        data.append(pd.read_csv(DATA_PATH + 'training_setB/' + file))
        print("  ", c, end='\r')
        c += 1
    print("  ", c)
    
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
    print("  ", c)
    for file in file_listB:
        data.append(pd.read_csv(DATA_PATH + 'training_setB/' + file))
        print("  ", c, end='\r')
        c += 1
    print("  ", c)

    # Concatenate all DataFrames in the list into a single DataFrame
    # dataset = pd.concat(data, ignore_index=True)

    print("Putting data into numpy array...")
    dataset = np.empty(len(data), dtype=object)
    for i in range(len(data)):
        dataset[i] = np.array(data[i])
    print("Done")

    return dataset

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
    dataset = get_dataset_as_np()  # Get the dataset

    ppca = PPCA()
    ppca.fit(data=dataset, d=2, verbose=True)  # Fit PPCA, d is the target dimensionality

    # The 'W' matrix contains the loadings for the principal components
    loadings = ppca.W
    print("Loadings (W matrix):\n", loadings)

    # To understand the contribution of each original feature to the principal components,
    # you can examine the absolute values of the loadings. Higher values indicate a stronger
    # contribution of that feature to the component.
    print("Absolute loadings:\n", np.abs(loadings))

    # If you want to rank features by their contribution to a specific component, you can do so:
    component_number = 0  # Change this based on the component you're interested in
    feature_importance = np.abs(loadings[:, component_number])
    feature_ranking = np.argsort(feature_importance)[::-1]

    print(f"Feature importance ranking for component {component_number}:\n", feature_ranking)

    feature_names = ['Feature1', 'Feature2', 'Feature3', ...]  # Fill this with your actual feature names
    
    # Plotting the loadings
    plot_loadings(ppca.W, feature_names)