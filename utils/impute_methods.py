from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

def impute_missing_values_knn(data, n_neighbors=5):
    """
    Imputes missing values using the KNN imputer.

    Parameters:
    - data (pd.DataFrame): The data frame containing the column values.
    - n_neighbors (int): Number of neighboring samples to use for imputation.

    Returns:
    - pd.DataFrame: DataFrame with missing values imputed.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(data)
    return pd.DataFrame(imputed_data, columns=data.columns)

def impute_linear_interpolation(data, column_name):
    """
    Imputes missing values using linear interpolation.

    Parameters:
    - data (pd.DataFrame): The data frame containing the column values.
    - column_name (str): The name of the column to impute.

    Returns:
    - pd.DataFrame: DataFrame with missing values imputed using linear interpolation.
    """
    imputed_data = data
    imputed_data[column_name] = data[column_name].interpolate(method='linear')
    return imputed_data

def impute_forward_fill_last_recorded(data, column_name, global_mean=None):
    """
    Forward fills missing values in a DataFrame column from the last recorded value,
    replacing NaN values at the beginning of the column with the mean of the rest of the data.
    
    Parameters:
        data (DataFrame): The data frame containing the column values.
        column_name (str): Name of the column to forward fill.
    
    Returns:
        DataFrame: DataFrame with missing values forward filled in the specified column.
    """

    imputed_data = data
    
    # Replace NaN values at the beginning with the mean of the rest of the data
    if pd.isnull(data[column_name].iloc[0]):
        imputed_data[column_name].iloc[0] = data[column_name].mean()
    
    # Forward fill missing values from the last recorded value
    imputed_data[column_name] = data[column_name].ffill()

    # If the first value is still NaN then replace with mean of the dataset
    if global_mean != None:
        if pd.isnull(data[column_name].iloc[0]):
            imputed_data[column_name] = global_mean
    
    return imputed_data


def impute_all(files, column_names):
    """
    Forward fills missing values in a DataFrame column from the last recorded value,
    replacing NaN values at the beginning of the column with the mean of the rest of the data.
    
    Parameters:
        files (List[DataFrame]): A list of DataFrames, each Dataframe represents a file
    
    Returns:
        DataFrame: DataFrame with missing values forward filled in the specified column.
    """

    df = pd.DataFrame(np.vstack(files), columns = column_names)
    global_means = df.mean(skipna=True)

    imputed_dataset = np.array([])

    for file in files:
        file = pd.DataFrame(file, columns=column_names)
        for column_name in column_names:
            file = impute_forward_fill_last_recorded(file, column_name, global_means[column_name])

        if imputed_dataset.size == 0:
            imputed_dataset = file.to_numpy()
        else:
            imputed_dataset = np.vstack([imputed_dataset, file.to_numpy()])

    return imputed_dataset
    