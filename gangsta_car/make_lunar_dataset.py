import os
import pickle
import random
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from tqdm import tqdm

waveforms_folder = os.path.abspath('./data/lunar/training/data/S12_GradeA')
catalogue_file = os.path.abspath('./data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv')

class WaveformDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            waveform = self.transform(waveform)

        return torch.tensor(waveform, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def load_waveform(file_path):
    try:
        waveform = pd.read_csv(file_path)
        return waveform['velocity(m/s)'].values
    except Exception as e:
        print(f"Error loading waveform {file_path}: {e}")
        return None

def process_labels(start, sequence_length=424):
    labels = np.zeros(sequence_length)
    labels[start] = 1
    return labels


def time_to_index(time_rel, sampling_rate, total_duration):
    return int((time_rel / total_duration) * sampling_rate)

def shift_waveform(waveform, index, sequence_length=424):
        # Calculate the random shift
        max_shift = sequence_length // 2
        random_shift = np.random.randint(-max_shift, max_shift + 1)

        start_idx = max(0, index - sequence_length // 2 + random_shift)
        end_idx = start_idx + sequence_length
        #Make sure that the waveform is not out of bounds
        if end_idx > waveform.shape[0]:
            end_idx = waveform.shape[0]
            start_idx = end_idx - sequence_length
        return waveform[start_idx:end_idx], start_idx

def load_waveforms_and_labels(waveforms_folder, catalogue_file, norm_percentile, sequence_length=424):
    waveforms = []
    labels = []

    # Load the catalogue
    catalogue_df = pd.read_csv(catalogue_file)

    # Load all waveforms and labels
    for file_name in tqdm(os.listdir(waveforms_folder), desc="Processing dataset"):
        if not file_name.endswith('.csv'):
            continue
        row_file_name = file_name.split('.csv')[0]
        # Find the corresponding row in the catalogue
        label_row = catalogue_df[catalogue_df['filename'] == row_file_name]

        if label_row.empty:
            continue

        time_rel = label_row['time_rel(sec)'].values[0]
        sampling_rate = 52 / 8  # 52 samples every 8 seconds
        total_duration = 64

        index = time_to_index(time_rel, sampling_rate, total_duration)

        waveform = load_waveform(os.path.join(waveforms_folder, f"{file_name}"))
        if waveform is not None:
            # Normalize max abs
            max_abs = np.percentile(np.abs(waveform), norm_percentile * 100)
            waveform = waveform / max_abs

            # Clip values to be within the range [-1, 1]
            waveform = np.clip(waveform, -1, 1)
            sampled_waveform, start_idx = shift_waveform(waveform, index, sequence_length)
            # Ensure the sampled waveform has the correct length
            if sampled_waveform.shape[0] == sequence_length:
                waveforms.append(sampled_waveform)
                labels.append(process_labels(index - start_idx, sequence_length))

    waveforms = np.array(waveforms)
    labels = np.array(labels)


    return waveforms, labels

def plot_picking_predictions(model, test_loader, device, num_samples=10):
    model.eval()
    samples = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.unsqueeze(-1))
            targets = targets.unsqueeze(2)
            samples.append((inputs.cpu(), targets.cpu(), outputs.cpu()))
            if len(samples) >= num_samples:
                break
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2 * num_samples))
    if num_samples == 1:
        axes = [axes]
    for i, (inputs, targets, outputs) in enumerate(samples[:num_samples]):
        if targets.dim() == 2:
            axes[i].plot(targets[0, :].numpy(), label='Actual', linestyle='--')
            axes[i].plot(outputs[0, :, 0].numpy(), label='Predicted', linestyle=':')
        elif targets.dim() == 3:
            axes[i].plot(targets[0, :, 0].numpy(), label='Actual', linestyle='--')
            axes[i].plot(outputs[0, :, 0].numpy(), label='Predicted', linestyle=':')
        axes[i].set_title(f"Sample {i+1}")
        axes[i].legend()
        axes[i].set_ylim(0, 1)
    plt.tight_layout()
    plt.show()

#Main processing function
def process_data( percentile=0.998 ): #["_N.csv", "_E.csv", "_Z.csv"]
    #Load the dataset
    waveforms, labels = load_waveforms_and_labels(waveforms_folder=waveforms_folder,
                                                  catalogue_file=catalogue_file,
                                                  norm_percentile=percentile,)

    #Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(waveforms, labels, test_size=0.5, random_state=None)
    # Save the split datasets
    with open('lunar_train_test_split.pkl', 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)

    #Create the dataset and dataloader
    train_dataset = WaveformDataset(X_train, y_train)
    test_dataset = WaveformDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

# process_data()