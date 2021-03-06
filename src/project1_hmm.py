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

def load_data(set_name):
    data_dir = '../data/%s' % (set_name,)
    data_file_path = data_dir+'/%s.pkl' % (set_name,)
    seq_file_path = data_dir+'/%s_seq.pkl' % (set_name,)

    if not os.path.isfile(data_file_path):
        print "%s data pickle isn't exist... wait" % (set_name,)
        data, seq_length = chunking_seq(data_dir)
        with open(data_file_path, 'wb+') as f:
            cPickle.dump(data, f)
        with open(seq_file_path, 'wb+') as f:
            cPickle.dump(seq_length, f)
    else:
        print "%s data pickle exist" % (set_name,)
        with open(data_file_path, 'rb')as f:
            data = cPickle.load(f)
        with open(seq_file_path, 'rb') as f:
            seq_length = cPickle.load(f)
    return data, seq_length

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
    data_file_path = data_dir+'/%s_feature.pkl' % (set_name,)
    seq_file_path = data_dir+'/%s_seq_feature.pkl' % (set_name,)
    if not os.path.isfile(data_file_path):
        print "%s feature pickle isn't exist... wait" % (set_name,)

        new_data_dic = {}
        new_seq_length_dic = {}

        transportation_modes = ['train', 'car', 'bus', 'walk', 'bike']
        for mode in transportation_modes:
            print "feature_engineering... transportation_modes mode : %s" % mode
            data = data_dic[mode]
            seq_length = seq_length_dic[mode]

            index = []
            session = 1
            for l in seq_length:
                index += [session] * l
                session += 1

            df_data = pd.DataFrame(data, columns=['lat', 'lng', 'datetime', 'alt'])
            df_data.index = index
            df_data['unixtime'] = [time.mktime(d.to_pydatetime().timetuple()) for d in df_data['datetime']]

            df_data['lat-1'] = df_data['lat'].shift(-1)
            df_data['lng-1'] = df_data['lng'].shift(-1)
            df_data['unixtime-1'] = df_data['unixtime'].shift(-1)

            # calucate distance, time_delta and velocity
            df_data['distance'] = [calculate_distance(d[1]['lng'], d[1]['lat'], d[1]['lng-1'], d[1]['lat-1'])for d in df_data.iterrows()]
            df_data['time_delta'] = df_data['unixtime-1'] - df_data['unixtime']
            df_data['velocity'] = df_data['distance'] / df_data['time_delta']  # m/s

            # calculate accelerometer
            df_data['velocity-1'] = df_data['velocity'].shift(-1)
            df_data['velocity_delta'] = df_data['velocity-1'] - df_data['velocity']
            df_data['acc'] = df_data['velocity_delta'] / df_data['time_delta']

            # weekend
            df_data['weekdays'] = [10 if d.weekday() > 4 else 1 for d in df_data['datetime']]

            # rushhour
            df_data['rushhour'] = [10 if (d.time().hour > 6 and d.time().hour < 10) or (d.time().hour > 16 and d.time().hour < 20) else 1 for d in df_data['datetime']]
            # df_data['hours'] = [d.time().hour  for d in df_data['datetime']]

            # print df_data

            # change dataframe to dic
            session = 1
            data_list = []
            seq_list = []
            for l in seq_length:
                # remove last two rows per index(session)
                try:
                    feature_set = ['velocity', 'acc', 'weekdays', 'rushhour']
                    df_features = df_data[feature_set]
                    df_feature_by_session = df_features.loc[session]
                    dropped_df = df_feature_by_session.replace([np.inf, -np.inf], np.nan).dropna(how="any")
                    removed_length = dropped_df.shape[0]
                    # print removed_length
                    if removed_length-2 <= 0:
                        session += 1
                        continue

                    data_list += dropped_df.as_matrix()[:-2, :].tolist()
                    seq_list.append(removed_length-2)
                except IndexError:
                    pass

                session += 1

            new_data_dic[mode] = data_list
            new_seq_length_dic[mode] = seq_list

            # # debugging
            # print len(data_list)
            # print sum(seq_list)
            # for l in seq_list:
            #    if l == 0:
            #        print "length 0 is found"

        # save features to pickle
        with open(data_file_path, 'wb+') as f:
            cPickle.dump(new_data_dic, f)
        with open(seq_file_path, 'wb+') as f:
            cPickle.dump(new_seq_length_dic, f)

    else:
        # load features from pickle file
        print "%s feature pickle exist" % (set_name,)
        with open(data_file_path, 'rb')as f:
            new_data_dic = cPickle.load(f)
        with open(seq_file_path, 'rb') as f:
            new_seq_length_dic = cPickle.load(f)

    return new_data_dic, new_seq_length_dic

def train_and_validate():
    start_time = time.time()

    train_data, train_seq_length = load_data(set_name="train")
    test_data, test_seq_length = load_data(set_name="test")

    print("--- data is loaded : %s seconds ---" % (time.time() - start_time))
    start_time2 = time.time()

    # feature engineering
    # you can modify utility module
    # TODO
    train_data, train_seq_length = feature_engineering(train_data, train_seq_length, "train")
    test_data, test_seq_length = feature_engineering(test_data, test_seq_length, "test")

    print("--- features are made : %s seconds ---" % (time.time() - start_time2))
    start_time2_5 = time.time()

    # learning process
    # TODO
    print 'train'
    model_train = hmm.GaussianHMM(n_components=20, covariance_type='diag', algorithm='viterbi', n_iter=20).fit(train_data['train'], train_seq_length['train'])
    print 'car'
    model_car = hmm.GaussianHMM(n_components=20, covariance_type='diag', algorithm='viterbi', n_iter=20).fit(train_data['car'], train_seq_length['car'])
    print 'bus'
    model_bus = hmm.GaussianHMM(n_components=20, covariance_type='diag', algorithm='viterbi', n_iter=20).fit(train_data['bus'], train_seq_length['bus'])
    print 'walk'
    model_walk = hmm.GaussianHMM(n_components=20, covariance_type='diag', algorithm='viterbi', n_iter=20).fit(train_data['walk'], train_seq_length['walk'])
    print 'bike'
    model_bike = hmm.GaussianHMM(n_components=20, covariance_type='diag', algorithm='viterbi', n_iter=20).fit(train_data['bike'], train_seq_length['bike'])

    print("--- model is trained : %s seconds ---" % (time.time() - start_time2_5))
    start_time3 = time.time()

    # test process
    # TODO
    actual = []
    predict = []
    for idx, key in enumerate(sorted(test_data.keys())):
        print key
        seqs = test_data[key]
        seq_len = test_seq_length[key]

        summation = 0
        for i in xrange(len(seq_len)):
            seq = seqs[summation:(summation+seq_len[i])]
            ll_train = model_train.score(seq)
            ll_car = model_car.score(seq)
            ll_bus = model_bus.score(seq)
            ll_walk = model_walk.score(seq)
            ll_bike = model_bike.score(seq)

            pred = np.argmax([ll_bike, ll_bus, ll_car, ll_train, ll_walk])
            actual.append(idx)
            predict.append(pred)

            summation += seq_len[i]

    print "confusion_matrix : ", confusion_matrix(actual, predict)
    print "f1_score : ", f1_score(actual, predict, average=None)
    print "precision : ", precision_score(actual, predict, average=None)
    print "recall : ", recall_score(actual, predict, average=None)
    print "macro f1_score : ", f1_score(actual, predict, average='macro')

    print("--- test is done : %s seconds ---" % (time.time() - start_time3))
    print("--- total execution time :  %s seconds ---" % (time.time() - start_time))


def test_with_saved_model():
    pass

if __name__ == "__main__":
    train_and_validate()