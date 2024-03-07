from sklearn.impute import KNNImputer
import pandas as pd

def impute_missing_values_knn(data, n_neighbors=5):
    """
    Imputes missing values using the KNN imputer.

    Parameters:
    - data (pd.DataFrame): The data frame containing the heart rate values.
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
    - data (pd.DataFrame): The data frame containing the heart rate values with a time index.
    - column_name (str): The name of the column to impute.

    Returns:
    - pd.DataFrame: DataFrame with missing values imputed using linear interpolation.
    """
    data[column_name] = data[column_name].interpolate(method='linear')
    return data
