import pandas as pd
import numpy as np
from constants import sep_col, con_col

def feature_missing_information(patient_data, columns):
    # temp_data holds the information from the patient file as well as the features that will be calculated
    temp_data = np.array(patient_data)

    # Calculate 3 features for each column, 2 respective of the frequency of NaN values and 1 respective of the change in recorded values
    for column in columns:
        data = np.array(patient_data[column])
        nan_pos = np.where(~np.isnan(data))[0]
        
        # Measurement frequency sequence
        interval_f1 = data.copy()
        # Measurement time interval
        interval_f2 = data.copy()

        # If all the values are NaN
        if (len(nan_pos) == 0):
            interval_f1[:] = 0
            temp_data = np.column_stack((temp_data, interval_f1))
            interval_f2[:] = -1
            temp_data = np.column_stack((temp_data, interval_f2))
        else :
            # Puts number of measurements into temp_data
            interval_f1[: nan_pos[0]] = 0
            for p in range(len(nan_pos)-1):
                interval_f1[nan_pos[p]: nan_pos[p+1]] = p + 1
            interval_f1[nan_pos[-1] :] = len(nan_pos)
            temp_data = np.column_stack((temp_data, interval_f1))

            # Puts the frequency of measurements into temp_data
            interval_f2[:nan_pos[0]] = -1
            for q in range(len(nan_pos) - 1):
                length = nan_pos[q+1] - nan_pos[q]
                for l in range(length):
                    interval_f2[nan_pos[q] + l] = l

            length = len(patient_data) - nan_pos[-1]
            for l in range(length):
                interval_f2[nan_pos[-1] + l] = l
            temp_data = np.column_stack((temp_data, interval_f2))

        # Differential features
        # These capture the change in values that have been recorded (quite simply as well but it should be just fine)
        diff_f = data.copy()
        diff_f = diff_f.astype(float)
        if len(nan_pos) <= 1:
            diff_f[:] = np.NaN
            temp_data = np.column_stack((temp_data, diff_f))
        else:
            diff_f[:nan_pos[1]] = np.NaN
            for p in range(1, len(nan_pos)-1):
                diff_f[nan_pos[p] : nan_pos[p+1]] = data[nan_pos[p]] - data[nan_pos[p-1]]
            diff_f[nan_pos[-1]:] = data[nan_pos[-1]] - data[nan_pos[-2]]
            temp_data = np.column_stack((temp_data, diff_f))
    
    return temp_data

def feature_slide_window(patient_data, columns):
    
    window_size = 6
    features = {}
    
    for column in columns:
        series = patient_data[column]

        features[f'{column}_max'] = series.rolling(window=window_size, min_periods=1).max()
        features[f'{column}_min'] = series.rolling(window=window_size, min_periods=1).min()
        features[f'{column}_mean'] = series.rolling(window=window_size, min_periods=1).mean()
        features[f'{column}_median'] = series.rolling(window=window_size, min_periods=1).median()
        features[f'{column}_std'] = series.rolling(window=window_size, min_periods=1).std()
        
        # For calculating std dev of differences, use diff() then apply rolling std
        diff_std = series.diff().rolling(window=window_size, min_periods=1).std()
        features[f'{column}_diff_std'] = diff_std

    # Convert the dictionary of features into a DataFrame
    features_df = pd.DataFrame(features)
    
    return features_df

def features_score(patient_data):
    """
    Gives score assocciated with the patient data according to the scoring systems of NEWS, SOFA and qSOFA
    """
    
    scores = np.zeros((len(patient_data), 8))
    
    for ii in range(len(patient_data)):
        HR = patient_data[ii, 0]
        if HR == np.nan:
            HR_score = np.nan
        elif (HR <= 40) | (HR >= 131):
            HR_score = 3
        elif 111 <= HR <= 130:
            HR_score = 2
        elif (41 <= HR <= 50) | (91 <= HR <= 110):
            HR_score = 1
        else:
            HR_score = 0
        scores[ii, 0] = HR_score

        Temp = patient_data[ii, 2]
        if Temp == np.nan:
            Temp_score = np.nan
        elif Temp <= 35:
            Temp_score = 3
        elif Temp >= 39.1:
            Temp_score = 2
        elif (35.1 <= Temp <= 36.0) | (38.1 <= Temp <= 39.0):
            Temp_score = 1
        else:
            Temp_score = 0
        scores[ii, 1] = Temp_score

        Resp = patient_data[ii, 6]
        if Resp == np.nan:
            Resp_score = np.nan
        elif (Resp < 8) | (Resp > 25):
            Resp_score = 3
        elif 21 <= Resp <= 24:
            Resp_score = 2
        elif 9 <= Resp <= 11:
            Resp_score = 1
        else:
            Resp_score = 0
        scores[ii, 2] = Resp_score

        Creatinine = patient_data[ii, 19]
        if Creatinine == np.nan:
            Creatinine_score = np.nan
        elif Creatinine < 1.2:
            Creatinine_score = 0
        elif Creatinine < 2:
            Creatinine_score = 1
        elif Creatinine < 3.5:
            Creatinine_score = 2
        else:
            Creatinine_score = 3
        scores[ii, 3] = Creatinine_score

        MAP = patient_data[ii, 4]
        if MAP == np.nan:
            MAP_score = np.nan
        elif MAP >= 70:
            MAP_score = 0
        else:
            MAP_score = 1
        scores[ii, 4] = MAP_score

        SBP = patient_data[ii, 3]
        Resp = patient_data[ii, 6]
        if SBP + Resp == np.nan:
            qsofa = np.nan
        elif (SBP <= 100) & (Resp >= 22):
            qsofa = 1
        else:
            qsofa = 0
        scores[ii, 5] = qsofa

        Platelets = patient_data[ii, 30]
        if Platelets == np.nan:
            Platelets_score = np.nan
        elif Platelets <= 50:
            Platelets_score = 3
        elif Platelets <= 100:
            Platelets_score = 2
        elif Platelets <= 150:
            Platelets_score = 1
        else:
            Platelets_score = 0
        scores[ii, 6] = Platelets_score

        Bilirubin = patient_data[ii, 25]
        if Bilirubin == np.nan:
            Bilirubin_score = np.nan
        elif Bilirubin < 1.2:
            Bilirubin_score = 0
        elif Bilirubin < 2:
            Bilirubin_score = 1
        elif Bilirubin < 6:
            Bilirubin_score = 2
        else:
            Bilirubin_score = 3
        scores[ii, 7] = Bilirubin_score
        
    return scores

def extract_features(patient_data, columns_to_drop = []):
    # Get the column with Sepsis Label as it is not the same for each row (check documentation)
    labels = np.array(patient_data['SepsisLabel'])
    patient_data = patient_data.drop(columns=columns_to_drop)

    # Gets information from the missing variables 
    # This can be useful as it shows the clinical judgment, the test has not been ordered 
    #                              (probably a good decision we should take into account)
    temp_data = feature_missing_information(patient_data, sep_col + con_col)
    temp = pd.DataFrame(temp_data)
    # To complete the data use forward-filling strategy
    temp = temp.fillna(method='ffill')
    # These are also the first set of features
    # In this configutation 99 (66 + 33 or 3 per column) features to be precise
    # They are also time indifferent
    features_A = np.array(temp)
    # The team did not use DBP, not sure why, might investigate this
    # columns = ['HR', 'O2Sat', 'SBP', 'MAP', 'Resp', 'DBP']
    
    # six-hour slide window statistics of selected columns
    columns = ['HR', 'O2Sat', 'SBP', 'MAP', 'Resp']
    features_B = feature_slide_window(patient_data, columns)

    # Score features based according to NEWS, SOFA and qSOFA
    features_C = features_score(features_A)
    
    features = np.column_stack([features_A, features_B, features_C])
    
    return features, labels

# Data Pre-processing
def preprecess_data(dataset, patient_id_map = None):
    frames_features = []
    frames_labels = []
    
    for patient_id in set(dataset.index.get_level_values(0)):
        if patient_id_map is not None:
            print(f"Processing data for patient ID: {patient_id}, File: {patient_id_map[patient_id]}", end='\r')
        
        patient_data = dataset.loc[patient_id]
    
        features, labels = extract_features(patient_data)
        features = pd.DataFrame(features)
        labels = pd.DataFrame(labels)
    
        frames_features.append(features)
        frames_labels.append(labels)

    data_features = np.array(pd.concat(frames_features))
    data_labels = (np.array(pd.concat(frames_labels)))[:, 0]
    
    # Randomly shuffle the data
    index = [i for i in range(len(data_labels))]
    np.random.shuffle(index)
    data_features = data_features[index]
    data_labels = data_labels[index]
    
    return data_features, data_labels

if __name__ == '__main__':
    print(sep_col+con_col)