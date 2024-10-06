import os
import pandas as pd
from scipy.signal import butter, filtfilt
import numpy as np
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import matplotlib.pyplot as plt
import heapq


class FilterAlgorithm:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def local_maxima(self, velocity_data, sampling_rate):
        window_len = int(600 * sampling_rate)
        num_max = 20
        grads = {}
        for i in range(0, len(velocity_data) - window_len, window_len):
            maxx = np.max(velocity_data[i:i + window_len]) - np.mean(velocity_data)
            grads[maxx] = i
        max_keys = heapq.nlargest(num_max, grads.keys())
        lst = [grads[key] for key in max_keys]
        triggers = [[i - window_len, i + window_len] for i in lst]
        return triggers

    def process_waveform(self, filename, time_abs='time_abs(%Y-%m-%dT%H:%M:%S.%f)', time_rel='time_rel(sec)',
                         velocity='velocity(m/s)'):
        input_file = os.path.join(self.data_path, filename)
        if not os.path.exists(input_file):
            print(f"Warning: waveform file {input_file} does not exist.")
            return np.array([])
        df = pd.read_csv(input_file)
        velocity_data = df[velocity].values
        time_data = df[time_rel].values

        sampling_rate = 1 / (df[time_rel][1] - df[time_rel][0])
        print(f"Sampling rate: {sampling_rate} Hz")

        triggers = np.array(self.local_maxima(velocity_data, sampling_rate))
        # print("finished")
        return np.array(triggers)

    def convert_to_batches_triggers(self, batch_size, filename):
        triggers = self.process_waveform(filename)
        print(triggers)
        batched_triggers = []
        for i in range(0, len(triggers)):
            batches = []
            for j in range(triggers[i][0], triggers[i][1], batch_size):
                batches.append([j, j + batch_size])
            batched_triggers.append(batches)
        return batched_triggers
