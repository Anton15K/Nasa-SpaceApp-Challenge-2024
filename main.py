import os
import numpy as np
import pandas as pd
import torch
from Data_Filter_Algorithm import FilterAlgorithm
from gangsta_car.utils.model_instances import PickerModel, DetectorModel
from gangsta_car.utils.preprocess import preprocess
from matplotlib import pyplot as plt

def change_file_to_batches(data_path, filename, batched_triggers, batch_size=600):
    """
    Splits the velocity data into batches based on trigger indices.

    Args:
        data_path (str): Path to the data directory.
        filename (str): Name of the CSV file.
        batched_triggers (list): List of batched trigger indices.
        batch_size (int): Size of each batch (default is 600).

    Returns:
        list: Nested list of velocity batches.
    """
    # Read the CSV file
    data = pd.read_csv(os.path.join(data_path, filename))
    velocity = data["velocity(m/s)"].values

    batched_velocity = []
    for window_idx, window_triggers in enumerate(batched_triggers):
        batches = []
        for trigger_idx, (start_idx, end_idx) in enumerate(window_triggers):
            # Extract the chunk based on start and end indices
            batch = velocity[start_idx:end_idx]
            # If the chunk is smaller than batch_size, pad it with zeros
            if len(batch) < batch_size:
                padding = batch_size - len(batch)
                batch = np.pad(batch, (0, padding), 'constant')
            batches.append(batch)
        batched_velocity.append(batches)
    return batched_velocity



def perform_detection_and_picking(detection_model, picker_model, batched_velocity, batched_triggers):
    """
    Args:
        detection_model (DetectorModel): The convolutional neural network model for detection.
        picker_model (PickerModel): The BiLSTM model for precise event picking.
        batched_velocity (list): Nested list of velocity batches.
        batched_triggers (list): List of batched trigger indices.

    Returns:
        list: Indices of detected events in the initial waveform.
    """
    event_indices = []

    # Iterate over each window
    for window_idx in range(len(batched_velocity)):
        window_chunks = batched_velocity[window_idx]  # List of chunks in this window
        window_chunk_triggers = batched_triggers[window_idx]  # Corresponding triggers

        # Iterate over each chunk in the window
        for chunk_idx, chunk in enumerate(window_chunks):
            # Preprocess the chunk
            chunk_preprocessed = preprocess(chunk, 0.998)  # Normalize the chunk

            # Prepare tensor for detection model: [1, 1, 600]
            chunk_tensor_det = torch.tensor(chunk_preprocessed).unsqueeze(0).unsqueeze(2).float()

            # Run the detection model (ConvNet)
            prediction = detection_model.predict(chunk_tensor_det)

            if prediction:
                # Event detected, use the picker model to find the precise location

                # Prepare tensor for picker model: [1, 600, 1]
                chunk_tensor_pick = torch.tensor(chunk_preprocessed).unsqueeze(0).unsqueeze(2).float()

                # Run the picker model (BiLSTM)
                event_sample_in_chunk = picker_model.predict(chunk_tensor_pick)  # Returns int index

                # Calculate the absolute index in the initial waveform
                absolute_index = window_chunk_triggers[chunk_idx][0] + event_sample_in_chunk

                # Append the absolute index to the list
                event_indices.append(absolute_index)

                # Break after the first detected event in the window
                break

    return event_indices

def main():
    data_path = "test_data/lunar"
    filename = "xa.s15.00.mhz.1973-04-04HR00_evid00098.csv"
    batch_size = 600  # Size of chunks within each window
    window_size = 2400  # Size of windows

    algo = FilterAlgorithm(data_path)

    # Get batched triggers
    batched_triggers = algo.convert_to_batches_triggers_indexes(batch_size=batch_size, filename=filename)
    # The above method should return a list where each element corresponds to a window, containing multiple [start, end] indices for 600-sample chunks

    # Get batched velocity data (windows split into chunks of 600 samples)
    batched_velocity = change_file_to_batches(data_path, filename, batched_triggers, batch_size=batch_size)


    detection_model = DetectorModel("gangsta_car/models", "signal_detection_1ch.pth", "cpu")
    picker_model = PickerModel("gangsta_car/models", "1ch_single_var3.pth", "cpu")

    # Perform detection and picking
    event_indices = perform_detection_and_picking(detection_model, picker_model, batched_velocity, batched_triggers)
    # print("Detected event indices in the initial waveform:")
    # for index in event_indices:
    #     print(index)
    event_indices.sort()

    ans = []

    file = pd.read_csv(data_path + "/" + filename)
    rel_time = file["time_rel(sec)"].values
    for i in range(0, len(event_indices)):
        ans.append(rel_time[event_indices[i]])
    print(ans)


    dataframe = pd.DataFrame({
            'filename': [filename] * len(ans),
            'time_rel(sec)': ans
        })


    dataframe.to_csv(f"results/lunar/{filename}", index=False, header=True)

if __name__ == "__main__":
    main()
