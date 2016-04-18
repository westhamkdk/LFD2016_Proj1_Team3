import numpy as np
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from utility import chunking_seq
import time
import os
import cPickle

def load_data(set_name):
    data_dir = '../data/%s' %(set_name,)
    data_file_path = data_dir+'/%s.pkl'%(set_name,)
    seq_file_path = data_dir+'/%s_seq.pkl'%(set_name,)

    if not os.path.isfile(data_file_path):
        print "%s data pickle isn't exist... wait" %(set_name,)
        data, seq_length = chunking_seq(data_dir)
        with open(data_file_path, 'wb+') as f:
            cPickle.dump(data, f)
        with open(seq_file_path, 'wb+') as f:
            cPickle.dump(seq_length, f)
    else:
        print "%s data pickle exist" %(set_name,)
        with open(data_file_path, 'rb')as f:
            data = cPickle.load(f)
        with open(seq_file_path, 'rb') as f:
            seq_length = cPickle.load(f)
    return data, seq_length

def feature_engineering(data, seq_length):
    # print data['train']
    # print seq_length['train']
    return data, seq_length


def train_and_validate():
    start_time = time.time()

    train_data, train_seq_length = load_data(set_name="train")
    test_data, test_seq_length = load_data(set_name="test")

    print("--- data is loaded : %s seconds ---" % (time.time() - start_time))
    start_time2 = time.time()

    # feature engineering
    # you can modify utility module
    # TODO
    # train_data, train_seq_length = feature_engineering(train_data, train_seq_length)
    test_data, test_seq_length = feature_engineering(test_data, test_seq_length)

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
