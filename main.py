from Data_Filter_Algorithm import FilterAlgorithm
import numpy as np
import pandas as pd
from gangsta_car.utils.model_instances import PickerModel, DetectorModel
from gangsta_car.utils.preprocess import preprocess
import torch
from matplotlib import pyplot as plt

def change_file_to_batches(data_path, filename, batched_triggers):
    data = pd.read_csv(data_path + "/" + filename)
    velocity = data["velocity(m/s)"].values

    batched_velocity = []

    for i in range(0, len(batched_triggers)):
        batches = []
        for j in range(0, len(batched_triggers[i])):
            batch = velocity[batched_triggers[i][j][0]:batched_triggers[i][j][1]]
            batches.append(batch)
        batched_velocity.append(batches)
    return batched_velocity

def get_total_batch(batched_velocity : list):
    total_batch = []
    for j in range(0, len(batched_velocity)):
        for k in range(0, len(batched_velocity[j])):
            total_batch.append(batched_velocity[j][k])
    total_batch = preprocess(torch.tensor(total_batch), 0.998)
    return total_batch
def perform_detection(detection_model : DetectorModel, batched_velocity : list):
    picked_triggers = []
    for i in range(0, len(batched_velocity)):
        picked_indexes = []

        total_batch = get_total_batch(batched_velocity[i])

        plt.plot(total_batch)
        plt.show()

        picked_index = detection_model.predict(total_batch.unsqueeze(0).unsqueeze(2).float())
        picked_indexes.append(picked_index)
        picked_triggers.append(picked_indexes)
    return picked_triggers


def perform_detection_3_layers(detection_model : DetectorModel, batched_velocity : list):
    picked_triggers = []
    for i in range(0, len(batched_velocity)):
        picked_indexes = []

        total_batch = get_total_batch(batched_velocity[i])

        plt.plot(total_batch)
        plt.show()
        x = total_batch.repeat(3, 1)
        print(x.shape)
        #picked_index = detection_model.predict(total_batch.unsqueeze(0).unsqueeze(2).float())
        #picked_indexes.append(picked_index)
        #picked_triggers.append(picked_indexes)
    return picked_triggers









def main():
    data_path = "sample_data"
    algo = FilterAlgorithm(data_path)
    filename = "test_data.csv"
    batch_size = 600
    batched_triggers = algo.convert_to_batches_triggers_indexes(batch_size, filename)
    batched_velocity = change_file_to_batches(data_path, filename, batched_triggers)
    detection_model = DetectorModel("gangsta_car/models", "signal_detection.pth", "cpu")

    # for i in range(0, len(batched_velocity)):
    #     total_batch = get_total_batch(batched_velocity[i])
    #     plt.plot(total_batch)
    #     plt.show()
    #     plt.close()
    picked_triggers = perform_detection_3_layers(detection_model, batched_velocity)
    print(picked_triggers)


if __name__ == "__main__":
    main()