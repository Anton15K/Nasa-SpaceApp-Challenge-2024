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

waveforms_folder = './data/lunar/training/data/S12_GradeA'
catalogues_folder = './data/lunar/training/catalogs'

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


def downsample_waveform(waveform, factor = 10):
    try:
        #Downsample
        waveform["Velocity"] = pd.to_numeric(waveform["Velocity"], errors='coerce')

        downsampled_waveform = waveform["Velocity"].values.reshape(-1, factor).mean(axis=1)

        return downsampled_waveform
    except Exception as e:
        print(f"Error: {e}")

def delete_unlabelled_waveform(waveforms_folder, name):
    for channel in ["_N.csv", "_E.csv", "_Z.csv"]:
        file_path = os.path.join(waveforms_folder, f"{name}{channel}")
        if os.path.exists(file_path):
            os.remove(file_path)


def load_waveform(file_path):
    try:
        waveform = pd.read_csv(file_path)
        return downsample_waveform(waveform)
    except Exception as e:
        print(f"Error loading waveform {file_path}: {e}")
        return None

def process_labels(start, end, factor=10, sequence_length=1):
    start = int(float(start) * factor)
    # end = int(float(end) * factor)
    # Create a one-dimensional array of zeros with length = sequence_length
    labels = np.zeros(sequence_length)
    # labels[start:end+1] = 1
    labels[start] = 1
    return labels

def load_waveforms_and_labels(waveforms_folder, catalogues_folder, norm_percentile, channels,  labels_type='binary'): #, channels=["_N.csv", "_E.csv", "_Z.csv"], channels=["_Z.csv"]
    waveforms = []
    labels = []

    # Load all waveforms and labels
    for file_name in tqdm(os.listdir(waveforms_folder), desc="Processing dataset"):
        base_name = file_name[:-6]
        source_name = base_name.split('.')[0]

        # Load the catalogue for the station
        catalogue_file = os.path.join(catalogues_folder, f"catalogue_{source_name}.csv")
        if not os.path.exists(catalogue_file):
            print(f"Warning: Catalogue file {catalogue_file} does not exist.")
            continue

        catalogue_df = pd.read_csv(catalogue_file)
        label_row = catalogue_df[catalogue_df['filename'] == base_name]
        if len(label_row) == 0:
            delete_unlabelled_waveform(waveforms_folder, base_name)
            print(f"Warning: No matching label for {base_name} in {catalogue_file}")
            continue

        p_arrival = label_row['p_arrival_rel'].values[0]
        coda_end = label_row['coda_end_rel'].values[0]
        if labels_type == 'binary':
            if not pd.isna(p_arrival) or not pd.isna(coda_end):
                labels.append(1)
            else:
                labels.append(0)
        else:
            if pd.isna(p_arrival) or pd.isna(coda_end):
                continue # Skip samples with missing labels when making a dataset for the lstm model

            waveform_labels = process_labels(p_arrival, coda_end.strip('[[]]'), sequence_length=600)
            labels.append(waveform_labels)

        combined_waveform = []
        for channel in channels:
            channel_file = os.path.join(waveforms_folder, f"{base_name}{channel}")
            if not os.path.exists(channel_file):
                print(f"Warning: Channel {channel} missing for {file_name}")
                break

            waveform = load_waveform(channel_file)
            if waveform is not None:
                # Normalize waveforms
                max_abs = np.percentile(np.abs(waveform), norm_percentile*100)
                waveform = waveform / max_abs
                # Clip values to be within the range [-1, 1]
                waveform = np.clip(waveform, -1, 1)
                combined_waveform.append(waveform)

        waveforms.append(np.stack(combined_waveform, axis=-1))

    waveforms = np.array(waveforms)
    labels = np.array(labels)


    return waveforms, labels


def plot_waveforms(waveforms, labels, num_samples=5 ):
    # Check if labels are binary or contain p_arrival and coda_end
    if isinstance(labels[0], int):
        # Binary labels
        label_0_indices = [i for i, label in enumerate(labels) if label == 0]
        label_1_indices = [i for i, label in enumerate(labels) if label == 1]

        # Randomly select samples from each label category
        selected_indices = []
        num_samples_per_label = num_samples // 2

        if len(label_0_indices) < num_samples_per_label or len(label_1_indices) < num_samples_per_label:
            raise ValueError("Not enough samples for each label to plot")

        selected_indices.extend(random.sample(label_0_indices, num_samples_per_label))
        selected_indices.extend(random.sample(label_1_indices, num_samples_per_label))
    else:
        # p_arrival and coda_end labels
        selected_indices = random.sample(range(len(labels)), num_samples)

    # Plot the selected samples
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i, idx in enumerate(selected_indices):
        waveform = waveforms[idx]
        label = labels[idx]
        axes[i].plot(waveform)
        if isinstance(label, int):
            axes[i].set_title(f"Sample {i+1} - Label: {label}")
        else:
            p_arrival, coda_end = label
            axes[i].set_title(f"Sample {i+1} - P-arrival: {p_arrival}, Coda End: {coda_end}")
            ymin, ymax = axes[i].get_ylim()
            if not pd.isna(p_arrival):
                axes[i].vlines(p_arrival, ymin, ymax, color='b', linewidth=2, label='P-arrival')
            if not pd.isna(coda_end):
                axes[i].vlines(coda_end, ymin, ymax, color='aqua', linewidth=2, label='Coda End')
            axes[i].legend(loc='upper right', borderaxespad=0., prop={'weight': 'bold'})

        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Velocity")

    plt.tight_layout()
    plt.show()

def plot_picking_predictions(model, test_loader, device, num_samples=10):
    model.eval()
    samples = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
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
def process_data(labels_type, percentile=0.998,plot=False, channels=["_Z.csv"] ): #["_N.csv", "_E.csv", "_Z.csv"]
    #Load the dataset
    waveforms, labels = load_waveforms_and_labels(waveforms_folder=waveforms_folder,
                                                  catalogues_folder=catalogues_folder,
                                                  norm_percentile=percentile,
                                                  channels=channels,
                                                  labels_type=labels_type)

    if plot:
        plot_waveforms(waveforms, labels, num_samples=10)

    #Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(waveforms, labels, test_size=0.2, random_state=None)
    # Save the split datasets
    with open('train_test_split.pkl', 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)

    #Create the dataset and dataloader
    train_dataset = WaveformDataset(X_train, y_train)
    test_dataset = WaveformDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader