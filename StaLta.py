import os
import pandas as pd
from scipy.signal import butter, filtfilt
import numpy as np
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import matplotlib.pyplot as plt
data_path = './space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'
catalogs_path = './space_apps_2024_seismic_detection/data/lunar/training/catalogs'
def process_waveform(filename, time_abs='time_abs(%Y-%m-%dT%H:%M:%S.%f)', time_rel='time_rel(sec)', velocity='velocity(m/s)'):
    # Load the CSV file (replace 'your_file.csv' with the actual file path)
    input_file = os.path.join(data_path, filename)
    if not os.path.exists(input_file):
        print(f"Warning: waveform file {input_file} does not exist.")
        return np.array([])
    df = pd.read_csv(input_file)
    # Extract the seismic data (velocity) and time data
    velocity_data = df[velocity].values
    time_data = df[time_rel].values

    # Calculate the sampling rate (assuming uniform time intervals)
    sampling_rate = 1 / (df[time_rel][1] - df[time_rel][0])
    print(f"Sampling rate: {sampling_rate} Hz")

    sta_len = 120
    lta_len = 720
    # Apply classic STA/LTA
    cft = classic_sta_lta(velocity_data, int(sta_len * sampling_rate), int(lta_len * sampling_rate))
    #print(cft)
    # Plot the STA/LTA ratio

    threshold_on = 1.3
    threshold_off = 0.5
    #print(vel_mean)
    # Detect triggers based on the STA/LTA ratio
    triggers = np.array(trigger_onset(cft, threshold_on, threshold_off))
    triggers = triggers / sampling_rate
    expand_val = 60
    triggers[:, 0] -= expand_val
    print(triggers)
    # Print detected trigger points
    print(len(triggers))
    print((triggers[:, 1] - triggers[:, 0]).sum(axis=0))
    #print(triggers)

    # Plot waveform and mark detected events
    '''plt.figure()
    plt.plot(time_data, cft)
    plt.title('STA/LTA Ratio')
    plt.xlabel('Time (s)')
    plt.ylabel('STA/LTA Ratio')
    plt.show()

    # Set thresholds for event detection'''
    '''plt.figure()
    plt.plot(time_data, velocity_data, label="Waveform")
    for trigger in triggers:
        plt.axvline(time_data[trigger[0]], color='r', label="Trigger Onset")
    plt.title("Waveform with STA/LTA Trigger Onsets")
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.legend()
    plt.show()'''
    print("finished")
    return np.array(triggers)


def check_catalog(catalog_name):
    input_file = os.path.join(catalogs_path, catalog_name)
    if not os.path.exists(input_file):
        print(f"Warning: Catalogue file {input_file} does not exist.")
        return
    df_cat = pd.read_csv(input_file)
    filenames = df_cat["filename"].values
    print(filenames)
    seismic = df_cat["time_rel(sec)"].values
    res = 0
    for idx, filename in enumerate(filenames):
        filename += ".csv"
        print(f"{idx}: {filename}")
        triggers = process_waveform(filename)
        if len(triggers) == 0:
            continue
        event = seismic[idx,]
        lies_between = (triggers[:, 0] <= event) & (event <= triggers[:, 1])
        is_any_between = np.any(lies_between)
        print(is_any_between)
        if is_any_between:
            res += 1
    print(f'Result: {res} / {filenames.shape[0]}')

check_catalog('apollo12_catalog_GradeA_final.csv')
#process_waveform("xa.s12.00.mhz.1971-01-28HR00_evid00023.csv")