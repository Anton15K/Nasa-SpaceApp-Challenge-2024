import os
import numpy as np
import pandas as pd
import torch
from Data_Filter_Algorithm import FilterAlgorithm
from gangsta_car.utils.model_instances import PickerModel, DetectorModel
from gangsta_car.utils.preprocess import preprocess
from matplotlib import pyplot as plt

def change_file_to_batches(data_path, filename, batched_triggers):
    """
    Args:
        data_path (str): Path to the data directory.
        filename (str): Name of the CSV file.
        batched_triggers (list): List of batched trigger indices.

    Returns:
        list: Nested list of velocity batches.
    """
    data = pd.read_csv(os.path.join(data_path, filename))
    velocity = data["velocity(m/s)"].values

    batched_velocity = []
    for i in range(len(batched_triggers)):
        batches = []
        for j in range(len(batched_triggers[i])):
            start_idx, end_idx = batched_triggers[i][j]
            batch = velocity[start_idx:end_idx]
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

            chunk_preprocessed = preprocess(chunk, 0.998)  # Normalize the chunk
            chunk_tensor = torch.tensor(chunk_preprocessed).unsqueeze(0).unsqueeze(2).float()  # Shape: [1, 600, 1]

            # Run the detector
            prediction = detection_model.predict(chunk_tensor)

            if prediction == 1:
                picker_input = chunk_tensor
                event_probs = picker_model.predict(picker_input)  #[1, 600, 1]

                event_probs = event_probs.squeeze(2)  # Shape: [1, 600]
                event_sample_in_chunk = torch.argmax(event_probs, dim=1).item()  # Index within the chunk

                # Calculate the absolute index
                absolute_index = window_chunk_triggers[chunk_idx][0] + event_sample_in_chunk

                event_indices.append(absolute_index)
                break

    return event_indices

def main():
    data_path = "sample_data"
    filename = "test_data.csv"
    batch_size = 600  # Size of chunks within each window
    window_size = 2400  # Size of windows

    algo = FilterAlgorithm(data_path)

    batched_triggers = algo.convert_to_batches_triggers_indexes(window_size, filename) #2400
    batched_velocity = change_file_to_batches(data_path, filename, batched_triggers)   #600


    detection_model = DetectorModel("gangsta_car/models", "signal_detection.pth", "cuda")
    picker_model = PickerModel("gangsta_car/models", "event_picker.pth", "cuda")

    # Perform detection and picking
    event_indices = perform_detection_and_picking(detection_model, picker_model, batched_velocity, batched_triggers)

    print("Detected event indices in the initial waveform:")
    for index in event_indices:
        print(index)

if __name__ == "__main__":
    main()
