import datetime
import numpy as np
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from utility import chunking_seq
import time
import os
import cPickle
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import plotly.plotly as py
from scipy.stats.stats import pearsonr

def calculate_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    m = km*1000
    return m


def feature_engineering(data_dic, seq_length_dic, set_name):
    data_dir = '../data/%s' %(set_name,)
    data_file_path = data_dir+'/%s_feature.pkl'%(set_name,)
    seq_file_path = data_dir+'/%s_seq_feature.pkl'%(set_name,)
    if not os.path.isfile(data_file_path):
        print "%s feature pickle isn't exist... wait" %(set_name,)

        new_data_dic = {}
        new_seq_length_dic = {}

        transportation_modes = ['train', 'car', 'bus', 'walk', 'bike']
        for mode in transportation_modes:
            print "feature_engineering... transportation_modes mode : %s" %mode
            data = data_dic[mode]
            seq_length = seq_length_dic[mode]

            index = []
            session = 1
            for l in seq_length:
                index += [session] * l
                session += 1

            df_data = pd.DataFrame(data, columns=['lat', 'lng', 'datetime'])
            df_data.index = index
            df_data['unixtime'] = [time.mktime(d.to_pydatetime().timetuple()) for d in df_data['datetime']]

            df_data['lat-1'] = df_data['lat'].shift(-1)
            df_data['lng-1'] = df_data['lng'].shift(-1)
            df_data['unixtime-1'] = df_data['unixtime'].shift(-1)

            ## calucate distance, time_delta and velocity
            df_data['distance'] = [calculate_distance(d[1]['lng'], d[1]['lat'], d[1]['lng-1'], d[1]['lat-1'])for d in df_data.iterrows()]
            df_data['time_delta'] = df_data['unixtime-1'] - df_data['unixtime']
            df_data['velocity'] = df_data['distance'] / df_data['time_delta']  ## m/s

            ## calculate accelerometer
            df_data['velocity-1'] = df_data['velocity'].shift(-1)
            df_data['velocity_delta'] = df_data['velocity-1'] - df_data['velocity']
            df_data['acc'] = df_data['velocity_delta'] / df_data['time_delta']

            # print df_data

            ## change dataframe to dic
            session = 1
            data_list = []
            seq_list = []
            for l in seq_length:
                ## remove last two rows per index(session)
                try:
                    feature_set = ['velocity','acc']
                    df_features = df_data[feature_set]
                    df_feature_by_session = df_features.loc[session]
                    dropped_df = df_feature_by_session.replace([np.inf, -np.inf], np.nan).dropna(how="any")
                    removed_length = dropped_df.shape[0]
                    # print removed_length
                    if removed_length-2 <= 0:
                        session += 1
                        continue

                    data_list += dropped_df.as_matrix()[:-2,:].tolist()
                    seq_list.append(removed_length-2)
                except IndexError:
                    pass

                session += 1

            new_data_dic[mode] = data_list
            new_seq_length_dic[mode] = seq_list

        ## save features to pickle
        with open(data_file_path, 'wb+') as f:
            cPickle.dump(new_data_dic, f)
        with open(seq_file_path, 'wb+') as f:
            cPickle.dump(new_seq_length_dic, f)

    else:
        ## load features from pickle file
        print "%s feature pickle exist" %(set_name,)
        with open(data_file_path, 'rb')as f:
            new_data_dic = cPickle.load(f)
        with open(seq_file_path, 'rb') as f:
            new_seq_length_dic = cPickle.load(f)

    return new_data_dic, new_seq_length_dic


def plot_scatter():
    train_data, train_seq_length = None, None
    train_data, train_seq_length = feature_engineering(train_data, train_seq_length, "train")


    for mode in  ['train', 'car', 'bus', 'walk', 'bike']:
        print mode
        fig, ax = plt.subplots()
        data = train_data[mode]
        nd_data = np.asarray(data)
        ax.scatter(nd_data[:,0]-nd_data[:,0].mean(), nd_data[:,1]-nd_data[:,1].mean())
        ax.set_xlim([-100, 100])
        ax.set_ylim([-100, 100])
        print pearsonr(nd_data[:,0], nd_data[:,1])
        plt.show()


if __name__ == "__main__":
    plot_scatter()