import os
import sys
import random
import pandas as pd
import numpy as np
from scipy.linalg import toeplitz
from copy import copy
import matplotlib.pyplot as plt

# Geniuses that worked on hypertools did not update certain package and thus it produces warnings (they break jupyter lab)
import warnings
warnings.filterwarnings("ignore")

# Comment out if you don't want to see all of the values being printed (i.e. default)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

current_dir = os.getcwd()
utils_path = os.path.join(current_dir, 'utils')
utils_abs_path = os.path.abspath(utils_path)
if utils_abs_path not in sys.path:
    sys.path.append(utils_abs_path)

import utils.get_data as get_data
# from impute_methods import *
from utils.impute_methods import impute_linear_interpolation

# ___________________________________________________________________________________________________________________________________

import time
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.functional import pad

# ___________________________________________________________________________________________________________________________________

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerTimeSeries, self).__init__()

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x.transpose(0, 1))
        x = self.decoder(x)
        return x.squeeze(-1)

class PatientDataset(Dataset):
    def __init__(self, patient_ids, X, y, max_length, device):
        self.patient_ids = patient_ids
        self.device = device
        self.max_length = max_length
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, index):
        pid = self.patient_ids[index]
        patient_data = self.X.loc[pid]
        X_train, seq_length = prepare_patient_data(patient_data, self.max_length)
        y_train = torch.tensor(self.y.loc[pid].values, dtype=torch.float32)

        # Ensure y_train is appropriately padded or trimmed to match X_train's length
        if len(y_train) > self.max_length:
            y_train = y_train[:self.max_length]
        elif len(y_train) < self.max_length:
            y_train = pad(torch.tensor(y_train, dtype=torch.float32), (0, self.max_length - len(y_train)), value=0)
        
        return X_train, y_train, len(y_train)
        # return X_train, y_train, seq_length


def prepare_patient_data(patient_data, max_length):
        # Standardizing the data
        scaler = StandardScaler()
        features = scaler.fit_transform(patient_data)
        # Padding
        padded_features = np.zeros((max_length, features.shape[1]))
        sequence_length = min(max_length, features.shape[0])
        padded_features[:sequence_length] = features[:sequence_length]
        return torch.tensor(padded_features, dtype=torch.float32), sequence_length

def main():
    dataset, patient_id_map = get_data.get_dataset()

    # First lets experiment with only raw data 
    # We have to however impute NaN values since Neural Networks can't (natively) handle them

    columns_to_linearly_interpolate = [
        'HR', 'O2Sat', 'SBP', 'MAP', 'DBP', 'Resp'
    ]

    # Feel free to omit this (EXPERIMENTAL)
    # Normilize the dataset
    if True:
        # Check if multiindex_df is indeed a MultiIndex DataFrame
        if isinstance(dataset.index, pd.MultiIndex):
            # Exclude 'SepsisLabel' from normalization
            features_to_normalize = dataset.columns.difference(['SepsisLabel'])

            # Normalize each patient's data
            # This will apply z-score normalization per patient per feature, excluding 'SepsisLabel'
            normalized_data = dataset[features_to_normalize].groupby(level=0).transform(
                lambda x: (x - x.mean()) / x.std())

            # Optionally fill NaN values if they are created by division by zero in cases where std is zero
            normalized_data = normalized_data.fillna(0)

            # Merge normalized data with the 'SepsisLabel' column
            dataset = pd.concat([normalized_data, dataset['SepsisLabel']], axis=1)
        else:
            print("The dataframe does not have a MultiIndex as expected.")

    # Linear Interpolation
    print("Linearly interpolating:")
    for col in columns_to_linearly_interpolate:
        if col != 'SepsisLabel':  # Ensure we do not interpolate 'SepsisLabel'
            dataset = impute_linear_interpolation(dataset, col)
            print(col)
    print("Done")

    # ___________________________________________________________________________________________________________________________________

            
    # ___________________________________________________________________________________________________________________________________

    def add_nan_indicators(df):
        for column in df.columns:
            df[column + '_nan'] = df[column].isna().astype(int)
        return df

    def downsample(X, y):
        index_0 = np.where(y == 0)[0]
        index_1 = np.where(y == 1)[0]
        print(index_0, index_1)

        if len(index_0) > len(index_1):
            index_0 = np.random.choice(index_0, size=len(index_1), replace=False)

        balanced_indices = np.concatenate([index_0, index_1])
        np.random.shuffle(balanced_indices)

        x_balanced = X.iloc[balanced_indices]
        y_balanced = y.iloc[balanced_indices]

        return x_balanced, y_balanced


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training.")

    def compute_loss(outputs, targets, seq_lengths):
        # Transpose outputs to [batch_size, seq_length]
        outputs = outputs.transpose(0, 1).squeeze(-1)

        # Create a boolean mask for valid sequence elements
        max_seq_length = outputs.shape[1]
        mask = torch.arange(max_seq_length, device=outputs.device).expand(len(seq_lengths), max_seq_length) < torch.tensor(seq_lengths, device=outputs.device).unsqueeze(1)

        # Apply the mask to filter out invalid sequence elements
        masked_outputs = outputs[mask]
        masked_outputs = masked_outputs.float()

        # Flatten targets similarly, using the same mask
        masked_targets = targets[mask]
        masked_targets = masked_targets.float()

        # Calculate and return loss
        return criterion(masked_outputs, masked_targets)

    # ___________________________________________________________________________________________________________________________________


    X = dataset.drop('SepsisLabel', axis=1)
    X = add_nan_indicators(X)
    y = dataset['SepsisLabel']
    # just in case
    dataset *= 0

    print("Seeing if there are still any nan values or +/- infinities")
    # Just trying to fix some errors I got only on a GPU
    if X.isin([np.nan, np.inf, -np.inf]).any().any():
        print("Data contains NaN or infinite values. Handling...")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(method='ffill', inplace=True) 

    # Ensure no NaNs or infinities in the target variable as well
    if y.isin([np.nan, np.inf, -np.inf]).any():
        print("Target contains NaN or infinite values. Handling...")
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
        y.fillna(method='ffill', inplace=True)

    # Find the maximum sequence length for padding
    # Yes it's really high, 336, consider making it larger to accommodate actual test set
    max_length = X.groupby('patient_id').size().max()
    print("Max length (inputs will be padded to): ", max_length)

    patient_ids = X.index.get_level_values('patient_id').unique()
    train_ids, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)

    # Initialize model, criterion, and optimizer
    input_dim = X.shape[1]
    model = TransformerTimeSeries(input_dim=input_dim)

    classes = np.array([0, 1])
    class_weights = compute_class_weight('balanced', classes=classes, y=y.to_numpy())
    class_weights_tensor = torch.tensor(class_weights[1], dtype=torch.float32).to(device)  # Weight for the positive class

    # Before starting the training loop
    print(f"Class weights tensor: {class_weights_tensor}")

    # Initialize BCEWithLogitsLoss with pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ___________________________________________________________________________________________________________________________________

    # Training loop
    num_epochs = 10
    generator = torch.Generator(device="cuda")
    batch_size = 128
    num_workers = 4
    dataloader = DataLoader(PatientDataset(train_ids, X, y, max_length, device), batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator)
    
    print("Started Training")
    print(device)
    model.to(device)
    if torch.cuda.is_available():
        model.cuda()
    

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_preds, train_targets = [], []  # Reset predictions and targets at start of epoch

        start_time = time.time()
        total_batches = len(dataloader)

        for ii, (X_batch, y_batch, seq_lengths) in enumerate(dataloader):
            try:
                seq_lengths = torch.tensor(seq_lengths, device=device)
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)

                loss = compute_loss(outputs, y_batch, seq_lengths)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                valid_labels = torch.cat([y_batch[j][:seq_lengths[j]] for j in range(len(seq_lengths))])
                
                predicted_labels = (outputs.sigmoid() > 0.5).int()
                predicted_labels = predicted_labels.transpose(0, 1)
                # for i, length in enumerate(seq_lengths):
                #     train_correct += (predicted_labels[i][:length] == y_batch[i][:length]).sum().item()
                #     train_total += length
                for i, length in enumerate(seq_lengths):
                    train_correct += (predicted_labels[i][:length] == y_batch[i][:length]).sum().item()
                    train_total += length
                    train_preds.extend(predicted_labels[i][:length].tolist())  # Note the change here


                # train_preds.extend(predicted_labels.transpose(0, 1).tolist())
                train_targets.extend(valid_labels.tolist())

            except Exception as e:
                print(f"Error processing batch: {e}")
                break
            
            # Progress and time estimation
            end_batch = time.time()
            elapsed_time = end_batch - start_time
            time_per_batch = elapsed_time / (ii + 1)
            estimated_time_remaining = (total_batches - (ii + 1)) * time_per_batch
            
            print(f'Processed {ii+1}/{total_batches} batches, ' +
                f'Estimated time remaining: {estimated_time_remaining / 60:.2f} minutes', end='\r')

        unique_targets = set(train_targets)
        unique_preds = set(train_preds)

        # You can also add assertions to ensure only 0 and 1 are present
        assert unique_targets.issubset({0, 1}), "Targets contain more than two classes"
        assert unique_preds.issubset({0, 1}), "Predictions contain more than two classes"

        # Metrics calculation using the entire epoch's accumulated predictions and labels
        train_accuracy = 0
        if train_total > 0:
            train_accuracy = train_correct / train_total

        train_precision = precision_score(train_targets, train_preds, zero_division=0)
        train_recall = recall_score(train_targets, train_preds, zero_division=0)
        train_f1 = f1_score(train_targets, train_preds, zero_division=0)

        print(f'\nEpoch {epoch+1}, Avg Training Loss: {train_loss / total_batches:.4f}, ' +
            f'Training Metrics: Acc: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}')

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_preds, val_targets = [], []
        start_val_time = time.time()

        with torch.no_grad():
            for i, patient_id in enumerate(val_ids):
                start_batch = time.time()

                try:
                    patient_data = X.loc[patient_id]
                    X_val, sequence_length = prepare_patient_data(patient_data, max_length)
                    y_val = torch.tensor(y.loc[patient_id].values, dtype=torch.float32)

                    val_outputs = model(X_val.unsqueeze(0))
                    v_loss = criterion(val_outputs[:sequence_length].squeeze(), y_val[:sequence_length])
                    val_loss += v_loss.item()

                    val_predicted_labels = torch.round(torch.sigmoid(val_outputs[:sequence_length].squeeze()))
                    val_correct += (val_predicted_labels == y_val[:sequence_length]).sum().item()
                    val_total += sequence_length

                    val_preds.extend(val_predicted_labels.tolist())
                    val_targets.extend(y_val[:sequence_length].tolist())
                
                except Exception as e:
                    print(f"Error processing patient ID {patient_id}: {e}")

                # Progress and time estimation for validation
                end_batch = time.time()
                elapsed_time = end_batch - start_val_time
                batches_done = i + 1
                total_batches = len(val_ids)
                time_per_batch = elapsed_time / batches_done
                estimated_time_remaining = (total_batches - batches_done) * time_per_batch
                
                print(f'Validation: Processed {batches_done}/{total_batches} patients ({100.0 * batches_done / total_batches:.2f}%), ' +
                    f'Estimated time remaining: {estimated_time_remaining / 60:.2f} minutes', end='\r')

        val_accuracy = val_correct / val_total
        val_precision = precision_score(val_targets, val_preds)
        val_recall = recall_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds)

        # Print epoch summary                   
        print(f'Epoch {epoch+1}, Avg Training Loss: {train_loss / len(train_ids)}, Training Metrics: Acc: {train_accuracy}, Precision: {train_precision}, Recall: {train_recall}, F1: {train_f1}')
        print(f'Avg Validation Loss: {val_loss / len(val_ids)}, Validation Metrics: Acc: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}')

    save_model = True
    if save_model:
        torch.save(model, 'models/transformer/transformer_2l.pth')

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device('cuda')
    print(f"Using {device} for training.")
    main()