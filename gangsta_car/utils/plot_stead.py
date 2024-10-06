import os
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt

file_name = "../data/earth/raw/chunk5/chunk5.hdf5"
csv_file = "../data/earth/raw/chunk5/chunk5.csv"

# reading the csv file into a dataframe:
df = pd.read_csv(csv_file, low_memory=False)
print(f'total events in csv file: {len(df)}')
# filterering the dataframe
df = df[(df.trace_category == 'earthquake_local') & (df.source_distance_km <= 20) & (df.source_magnitude > 3)]
print(f'total events selected: {len(df)}')

# making a list of trace names for the selected data
ev_list = df['trace_name'].to_list()

# Check if the HDF5 file exists before opening
if os.path.exists(file_name):
    try:
        dtfl = h5py.File(file_name, 'r')
    except OSError as e:
        print(f"Error: Unable to open the file {file_name}. It may be corrupted.")
        print(e)
        exit(1)
else:
    print(f"Error: The file {file_name} does not exist.")
    exit(1)

# retrieving selected waveforms from the hdf5 file:
dtfl = h5py.File(file_name, 'r')
for c, evi in enumerate(ev_list):
    dataset = dtfl.get('data/'+str(evi))
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data = np.array(dataset)

    fig = plt.figure()
    ax = fig.add_subplot(311)
    plt.plot(data[:,0], 'k')
    plt.rcParams["figure.figsize"] = (8, 5)
    legend_properties = {'weight':'bold'}
    plt.tight_layout()
    ymin, ymax = ax.get_ylim()
    pl = plt.vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
    cl = plt.vlines(dataset.attrs['coda_end_sample'], ymin, ymax, color='aqua', linewidth=2, label='Coda End')
    plt.legend(handles=[pl, sl, cl], loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Amplitude counts', fontsize=12)
    ax.set_xticklabels([])

    ax = fig.add_subplot(312)
    plt.plot(data[:,1], 'k')
    plt.rcParams["figure.figsize"] = (8, 5)
    legend_properties = {'weight':'bold'}
    plt.tight_layout()
    ymin, ymax = ax.get_ylim()
    pl = plt.vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
    cl = plt.vlines(dataset.attrs['coda_end_sample'], ymin, ymax, color='aqua', linewidth=2, label='Coda End')
    plt.legend(handles=[pl, sl, cl], loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Amplitude counts', fontsize=12)
    ax.set_xticklabels([])

    ax = fig.add_subplot(313)
    plt.plot(data[:,2], 'k')
    plt.rcParams["figure.figsize"] = (8,5)
    legend_properties = {'weight':'bold'}
    plt.tight_layout()
    ymin, ymax = ax.get_ylim()
    pl = plt.vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
    cl = plt.vlines(dataset.attrs['coda_end_sample'], ymin, ymax, color='aqua', linewidth=2, label='Coda End')
    plt.legend(handles=[pl, sl, cl], loc = 'upper right', borderaxespad=0., prop=legend_properties)
    plt.ylabel('Amplitude counts', fontsize=12)
    ax.set_xticklabels([])
    plt.show()

    for at in dataset.attrs:
        print(at, dataset.attrs[at])

    inp = input("Press a key to plot the next waveform!")
    if inp == "r":
        continue