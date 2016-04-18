from os import listdir
from os.path import join
import time

import numpy as np
import pandas as pd


def chunking_seq(basedir):
    users = listdir(basedir)
    data = {'train': [], 'car':[], 'bus': [], 'walk': [], 'bike': []}
    seq_length = {'train': [], 'car': [], 'bus': [], 'walk': [], 'bike': []}
    for user in users:
        label_path = join(basedir, user, 'labels.txt')
        labels = pd.read_csv(label_path, sep='\t', parse_dates=[0, 1])

        trajectory_path = join(basedir, user, 'Trajectory')
        trajectories = listdir(trajectory_path)
        list.sort(trajectories)

        datetime_list = pd.DataFrame(columns=['start_time', 'end_time'])
        for idx, trajectory in enumerate(trajectories):
            raw_time = pd.read_csv(join(basedir, user, 'Trajectory', trajectory), skiprows=6, keep_date_col=True, delimiter=',', names=['lat', 'lng', '0', 'alt', 'day', 'date', 'time'])
            datetime = pd.to_datetime(raw_time['date'] + ' ' + raw_time['time'], format='%Y-%m-%d %H:%M:%S')
            row = {'start_time': np.min(datetime), 'end_time': np.max(datetime)}
            row = pd.DataFrame(row, columns=['start_time', 'end_time'], index=[idx])
            datetime_list = datetime_list.append(row)

        datetime_min = datetime_list['start_time']
        datetime_max = datetime_list['end_time']

        trajectories_date = [x.split('.')[0] for x in trajectories]
        trajectories_date = pd.to_datetime(trajectories_date, format='%Y%m%d%H%M%S')
        trajectories = np.array(trajectories)

        for index, row in labels.iterrows():
            start_time = row['Start Time']
            end_time = row['End Time']
            transport = row['Transportation Mode']

            # sequence_names = trajectories[np.array(datetime_max >= end_time)]
            sequence_names = trajectories[np.array(np.logical_and(datetime_min <= start_time, datetime_max-end_time >= '-60'))]

            # ttt = ttt + 1
            if len(sequence_names) > 0:
                for sequence_name in sequence_names:
                    sequence = pd.read_csv(join(basedir, user, 'Trajectory', sequence_name), skiprows=6, keep_date_col=True, delimiter=',', names=['lat', 'lng', '0', 'alt', 'day', 'date', 'time'])
                    sequence['datetime'] = pd.to_datetime(sequence['date'] + ' ' + sequence['time'], format='%Y-%m-%d %H:%M:%S')
                    sequence = sequence[np.logical_and(sequence['datetime'] >= start_time, sequence['datetime'] <= end_time)]
                    if sequence.shape[0] == 0:
                        continue


                    sequence = sequence.drop_duplicates(['datetime'])
                    # FIXME: select variable and feature engineering
                    sequence = sequence.as_matrix(['lat', 'lng', 'datetime'])
                    # TODO: feature engineering


                    if transport == 'train' or transport == 'light rail' or transport == 'subway':
                        data['train'].extend(sequence)
                        seq_length['train'].append(sequence.shape[0])
                    elif transport == 'car' or transport == 'taxi':
                        data['car'].extend(sequence)
                        seq_length['car'].append(sequence.shape[0])
                    elif transport == 'bus':
                        data['bus'].extend(sequence)
                        seq_length['bus'].append(sequence.shape[0])
                    elif transport == 'walk':
                        data['walk'].extend(sequence)
                        seq_length['walk'].append(sequence.shape[0])
                    elif transport == 'bike':
                        data['bike'].extend(sequence)
                        seq_length['bike'].append(sequence.shape[0])
                    else:
                        pass

    return data, seq_length