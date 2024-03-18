import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
'''
Initial read to understand the data - it reads the first 10 psv files from training_setA and imputates the NaN values for
a selected column, col_num. It uses linear regression to imputate the NaN values by taking non-NaN values either side of 
the NaN value and taking that value. It plots the original and modified points to see if the imputated values look correct

'''

folder_path = "/Users/Home/Library/CloudStorage/OneDrive-UniversityofBristol/3rd Year/Applied data science/training_setA"
file_list = os.listdir(folder_path)
psv_files = [file for file in file_list if file.endswith(".psv")]

dataframes = []
for file in psv_files[:10]:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, delimiter="|")
    dataframes.append(df)

# Choose which psv file to imputate, 0 to 9
df = dataframes[0]

# Choose which column to imputate
col_num = 8

column = df.iloc[:, col_num]


# Find indices of NaN values
nan_indices = column[column.isnull()].index

for index in nan_indices:

    # Find the nearest non-NaN values
    prev_index = column[:index].last_valid_index()
    next_index = column[index:].first_valid_index()

    # If there are no non-NaN values after find the nearest non-NaN values before
    if next_index is None:
        # Find the nearest non-NaN values before 
        prev_non_nan_index = column[:index].last_valid_index()
        prev_non_nan_index1 = column[:prev_non_nan_index].last_valid_index()

        # Get the corresponding x and y values for linear interpolation
        x = np.array([prev_non_nan_index1, prev_non_nan_index]).reshape(-1, 1)
        y = column.loc[[prev_non_nan_index1, prev_non_nan_index]]

        # Fit linear regression model for interpolation
        model = LinearRegression()
        model.fit(x, y)

        # Predict the missing value using interpolation
        predicted_value = model.predict([[index]])

        # Fill in the NaN value with the interpolated value
        column.loc[index] = predicted_value

    # If there are no non-NaN values before find the nearest non-NaN values after
    elif prev_index is None:
        # Find the nearest non-NaN values after
        next_non_nan_index = column[index:].first_valid_index()
        next_non_nan_index1 = column[next_non_nan_index+1:].first_valid_index()

        # Get the corresponding x and y values for linear interpolation
        x = np.array([next_non_nan_index, next_non_nan_index1]).reshape(-1, 1)
        y = column.loc[[next_non_nan_index, next_non_nan_index1]]

        # Fit linear regression model for interpolation
        model = LinearRegression()
        model.fit(x, y)

        # Predict the missing value using interpolation
        predicted_value = model.predict([[index]])

        # Fill in the NaN value with the interpolated value
        column.loc[index] = predicted_value


    else:
        # Get the corresponding x and y values for linear regression
        x = np.array([prev_index, next_index]).reshape(-1, 1)
        y = column.loc[[prev_index, next_index]]

        # Fit linear regression model
        model = LinearRegression()
        model.fit(x, y)

        # Predict the missing value
        predicted_value = model.predict([[index]])

        # Fill in the NaN value with the predicted value
        column.loc[index] = predicted_value

    if prev_index is None or next_index is None:
        continue


# Update the modified column in the dataframe
df.iloc[:, col_num] = column

# Plot the modified column
plt.scatter(df.index, df.iloc[:, col_num], color='blue')
plt.scatter(df.index[nan_indices], df.iloc[nan_indices, col_num], color='red')
plt.show()
