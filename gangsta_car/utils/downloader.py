import os
import pandas as pd
import h5py
import numpy as np
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client
from tqdm import tqdm
import multiprocessing
import random
from filelock import FileLock

def make_stream(dataset):
    data = np.array(dataset, dtype=dataset.dtype)
    # Only process Z channel
    channel = 'Z'
    i = 2  #['E', 'N', 'Z']

    tr = obspy.Trace(data=data[:, i])
    tr.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr.stats.delta = 0.01
    tr.stats.channel = dataset.attrs['receiver_type'] + channel
    tr.stats.station = dataset.attrs['receiver_code']
    tr.stats.network = dataset.attrs['network_code']

    return obspy.Stream([tr])

def save_waveform_to_csv(tr, evi, output_dir):
    relative_time = tr.times()
    output_file = os.path.join(output_dir, 'waveforms', f"{evi}_Z.csv")

    if os.path.exists(output_file):
        return

    lock_file = output_file + '.lock'
    with FileLock(lock_file):
        if os.path.exists(output_file):
            return
        with open(output_file, 'w') as f:
            waveform = tr.data
            np.savetxt(f, np.column_stack((relative_time, waveform)), delimiter=",", header="Time,Velocity", comments='')

def process_events_chunk(file_name, output_dir, ev_list):
    try:
        dtfl = h5py.File(file_name, 'r')
    except OSError as e:
        print(f"Error: Unable to open the file {file_name}. It may be corrupted.")
        print(e)
        return

    client = Client("IRIS")
    inventory_cache = {}

    for evi in ev_list:
        #check if the file already exists
        waveform_file = os.path.join(output_dir, 'waveforms', f"{evi}_Z.csv")
        if os.path.exists(waveform_file):
            continue  # Skip

        #check if event is catalogue
        station_code = evi.split('.')[0]
        catalogue_file = os.path.join(output_dir, 'catalogues', f'catalogue_{station_code}.csv')
        lock_path = catalogue_file + '.lock'

        with FileLock(lock_path):
            if os.path.exists(catalogue_file):
                existing_catalogue = pd.read_csv(catalogue_file)
                if evi in existing_catalogue['filename'].values:
                    continue  # Skip

        #Processing the event
        try:
            dataset = dtfl.get('data/' + str(evi))
            if dataset is None:
                print(f"Dataset {evi} not found in file.")
                continue

            #Save station inventoru
            station_key = (dataset.attrs['network_code'], dataset.attrs['receiver_code'])
            if station_key in inventory_cache:
                inventory = inventory_cache[station_key]
            else:
                inventory = client.get_stations(network=dataset.attrs['network_code'],
                                                station=dataset.attrs['receiver_code'],
                                                starttime=UTCDateTime(dataset.attrs['trace_start_time']),
                                                endtime=UTCDateTime(dataset.attrs['trace_start_time']) + 60,
                                                loc="*",
                                                channel="*",
                                                level="response")
                inventory_cache[station_key] = inventory

            st = make_stream(dataset)
            st.remove_response(inventory=inventory, output='VEL', plot=False)

            # Get arrival times
            p_arrival_sample = dataset.attrs.get('p_arrival_sample', None)
            s_arrival_sample = dataset.attrs.get('s_arrival_sample', None)
            coda_end_sample = dataset.attrs.get('coda_end_sample', None)

            tr = st[0]  # Only one trace, Z channel
            save_waveform_to_csv(tr, evi, output_dir)

            p_arrival_rel = p_arrival_sample * tr.stats.delta if p_arrival_sample is not None else None
            s_arrival_rel = s_arrival_sample * tr.stats.delta if s_arrival_sample is not None else None
            coda_end_rel = coda_end_sample * tr.stats.delta if coda_end_sample is not None else None

            catalogue_entry = pd.DataFrame([[evi, p_arrival_rel, s_arrival_rel, coda_end_rel]],
                                           columns=['filename', 'p_arrival_rel', 's_arrival_rel', 'coda_end_rel'])

            #Add to catalogue
            with FileLock(lock_path):
                if os.path.exists(catalogue_file):
                    existing_catalogue = pd.read_csv(catalogue_file)
                    combined_catalogue = pd.concat([existing_catalogue, catalogue_entry], ignore_index=True)
                else:
                    combined_catalogue = catalogue_entry
                combined_catalogue.drop_duplicates(subset='filename', inplace=True)

                # Save catalogue
                combined_catalogue.to_csv(catalogue_file, index=False)

        except Exception as e:
            print(f"Error processing event {evi}: {e}")
            continue

    dtfl.close()

def download_data(file_name, csv_file, output_dir, num_events, start_index=0, n_processes=4):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'waveforms'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'catalogues'), exist_ok=True)

    df = pd.read_csv(csv_file, low_memory=False)
    ev_list = df['trace_name'].to_list()[start_index:start_index + num_events]

    # Shuffle the event list
    random.shuffle(ev_list)

    chunk_size = len(ev_list) // n_processes
    chunks = [ev_list[i*chunk_size:(i+1)*chunk_size] for i in range(n_processes)]
    # Handle any leftover events
    if len(ev_list) % n_processes != 0:
        chunks[-1].extend(ev_list[n_processes*chunk_size:])

    # Prepare arguments for each process
    args_list = [(file_name, output_dir, chunk) for chunk in chunks]

    with multiprocessing.Pool(processes=n_processes) as pool:
        pool.starmap(process_events_chunk, args_list)

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'earth', 'raw', 'chunk5'))

    file_name = os.path.join(base_dir, 'chunk5.hdf5')
    csv_file = os.path.join(base_dir, 'chunk5.csv')
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'earth'))

    num_events = 10000
    start_index = 11200
    n_processes = 4

    download_data(file_name, csv_file, output_dir, num_events, start_index, n_processes)
