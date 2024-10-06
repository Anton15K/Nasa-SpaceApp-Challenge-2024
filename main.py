from Data_Filter_Algorithm import FilterAlgorithm
import numpy as np
import pandas as pd

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


def main():
    data_path = "sample_data"
    algo = FilterAlgorithm(data_path)
    filename = "test_data.csv"
    batch_size = 600
    batched_triggers = algo.convert_to_batches_triggers_indexes(batch_size, filename)
    print(len(batched_triggers))
    batched_velocity = change_file_to_batches(data_path, filename, batched_triggers)
    print(len(batched_velocity[0][2]))



if __name__ == "__main__":
    main()