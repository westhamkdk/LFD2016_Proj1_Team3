import numpy as np
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from utility import chunking_seq
import time
import os
import cPickle

def load_train_data():
    train_dir = '../data/train'
    data_file_path = train_dir+'/train.pkl'
    seq_file_path = train_dir+'/train_seq.pkl'

    if not os.path.isfile(data_file_path):
        print "train pickle isn't exist"
        train_data, train_seq_length = chunking_seq(train_dir)
        with open(data_file_path, 'wb+') as f:
            cPickle.dump(train_data, f)
        with open(seq_file_path, 'wb+') as f:
            cPickle.dump(train_seq_length, f)
    else:
        print "train pickle exist"
        with open(data_file_path, 'rb')as f:
            train_data = cPickle.load(f)
        with open(seq_file_path, 'rb') as f:
            train_seq_length = cPickle.load(f)
    return train_data, train_seq_length

# def load_test_data():
#      test_dir = '../data/test'
#      test_data, test_seq_length = chunking_seq(test_dir)
#      return test_data, test_seq_length

def load_test_data():
    test_dir = '../data/test'
    data_file_path = test_dir+'/test.pkl'
    seq_file_path = test_dir+'/train_seq.pkl'

    if not os.path.isfile(data_file_path):
        print "test pickle isn't exist"
        test_data, test_seq_length = chunking_seq(test_dir)
        with open(data_file_path, 'wb+') as f:
            cPickle.dump(test_data, f)
        with open(seq_file_path, 'wb+') as f:
            cPickle.dump(test_seq_length, f)
    else:
        print "test pickle exist"
        with open(data_file_path, 'rb')as f:
            test_data = cPickle.load(f)
        with open(seq_file_path, 'rb') as f:
            test_seq_length = cPickle.load(f)
    return test_data, test_seq_length


def train_and_validate():
    start_time = time.time()

    train_data, train_seq_length = load_train_data()
    test_data, test_seq_length = load_test_data()

    print("--- data is loaded : %s seconds ---" % (time.time() - start_time))
    start_time2 = time.time()

    # feature engineering
    # you can modify utility module
    # TODO

    # learning process
    # TODO
    model_train = hmm.GaussianHMM(n_components=5, n_iter=20).fit(train_data['train'], train_seq_length['train'])
    model_car = hmm.GaussianHMM(n_components=5, n_iter=20).fit(train_data['car'], train_seq_length['car'])
    model_bus = hmm.GaussianHMM(n_components=5, n_iter=20).fit(train_data['bus'], train_seq_length['bus'])
    model_walk = hmm.GaussianHMM(n_components=5, n_iter=20).fit(train_data['walk'], train_seq_length['walk'])
    model_bike = hmm.GaussianHMM(n_components=5, n_iter=20).fit(train_data['bike'], train_seq_length['bike'])

    print("--- model is trained : %s seconds ---" % (time.time() - start_time2))
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
    print "predicison : ", precision_score(actual, predict, average=None)
    print "recall : ", recall_score(actual, predict, average=None)
    print "macro f1_score : ", f1_score(actual, predict, average='macro')

    print("--- test is done : %s seconds ---" % (time.time() - start_time3))
    print("--- total execution time :  %s seconds ---" % (time.time() - start_time))


def test_with_saved_model():
    pass

if __name__ == "__main__":
    train_and_validate()
