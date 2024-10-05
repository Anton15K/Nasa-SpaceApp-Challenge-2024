import os
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client
from tqdm import tqdm
import time

def make_stream(dataset):
    data = np.array(dataset, dtype=dataset.dtype)
    channels = ['E', 'N', 'Z']
    traces = []

    for i, channel in enumerate(channels):
        tr = obspy.Trace(data=data[:, i])
        tr.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
        tr.stats.delta = 0.01
        tr.stats.channel = dataset.attrs['receiver_type'] + channel
        tr.stats.station = dataset.attrs['receiver_code']
        tr.stats.network = dataset.attrs['network_code']
        traces.append(tr)

    return obspy.Stream(traces)

def save_waveform_to_csv(tr, evi, channel, output_dir):
    relative_time = tr.times()
    output_file = os.path.join(output_dir, 'waveforms', f"{evi}_{channel}.csv")
    with open(output_file, 'a') as f:
        waveform=tr.data
        # waveform = downsample_waveform(tr.data)
        np.savetxt(f, np.column_stack((relative_time, waveform)), delimiter=",", header="Time,Velocity", comments='')

def store_catalogue(catalogue, output_dir, evi_prefix):
    catalogue_path = os.path.join(output_dir, 'catalogues', f'catalogue_{evi_prefix}.csv')
    new_catalogue_df = pd.DataFrame(catalogue, columns=['filename', 'p_arrival_rel', 's_arrival_rel', 'coda_end_rel'])

    # Check if the catalogue file already exists
    if os.path.exists(catalogue_path):
        existing_catalogue_df = pd.read_csv(catalogue_path)
        combined_catalogue_df = pd.concat([existing_catalogue_df, new_catalogue_df], ignore_index=True)
    else:
        combined_catalogue_df = new_catalogue_df
    # Save the combined data back to the file
    combined_catalogue_df.to_csv(catalogue_path, index=False)

def download_data(file_name, csv_file, output_dir, num_events, start_index=0):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file, low_memory=False)
    ev_list = df['trace_name'].to_list()[start_index:start_index + num_events]

    if not os.path.exists(file_name):
        print(f"Error: The file {file_name} does not exist.")
        return

    try:
        dtfl = h5py.File(file_name, 'r')
    except OSError as e:
        print(f"Error: Unable to open the file {file_name}. It may be corrupted.")
        print(e)
        return

    catalogue = []
    client = Client("IRIS")

    for c, evi in enumerate(tqdm(ev_list, desc="Processing events")):
        if c == 1:
            cat_path = os.path.join(output_dir, 'catalogues', f'catalogue_{ evi.split('.')[0]}.csv')
            if os.path.exists(cat_path):
                catalogue = pd.read_csv(cat_path).values.tolist()
        try:
            dataset = dtfl.get('data/' + str(evi))
            inventory = client.get_stations(network=dataset.attrs['network_code'],
                                            station=dataset.attrs['receiver_code'],
                                            starttime=UTCDateTime(dataset.attrs['trace_start_time']),
                                            endtime=UTCDateTime(dataset.attrs['trace_start_time']) + 60,
                                            loc="*",
                                            channel="*",
                                            level="response")

            st = make_stream(dataset)
            st.remove_response(inventory=inventory, output='VEL', plot=False) # correctly download dataset

            p_arrival_sample = dataset.attrs.get('p_arrival_sample', None)
            s_arrival_sample = dataset.attrs.get('s_arrival_sample', None)
            coda_end_sample = dataset.attrs.get('coda_end_sample', None)

            p_arrival_rel = p_arrival_sample * st[0].stats.delta if p_arrival_sample is not None else None
            s_arrival_rel = s_arrival_sample * st[0].stats.delta if s_arrival_sample is not None else None
            coda_end_rel = coda_end_sample * st[0].stats.delta if coda_end_sample is not None else None

            for i, tr in enumerate(st):
                save_waveform_to_csv(tr, evi, ['E', 'N', 'Z'][i], output_dir)

            catalogue.append([evi, p_arrival_rel, s_arrival_rel, coda_end_rel])
            # catalogue.append([evi, None, None, None])
        except Exception as e:
            print(f"Error processing event {evi}: {e}")
            continue

        if (c + 1) % 10 == 0:
            store_catalogue(catalogue, output_dir, evi.split('.')[0])
            time.sleep(5)

    store_catalogue(catalogue, output_dir, evi.split('.')[0])

file_name = "./data/earth/raw/chunk5/chunk5.hdf5"
csv_file = "./data/earth/raw/chunk5/chunk5.csv"
output_dir = "./data/earth"
num_events = 250
start_index = 10000
download_data(file_name, csv_file, output_dir, num_events, start_index)