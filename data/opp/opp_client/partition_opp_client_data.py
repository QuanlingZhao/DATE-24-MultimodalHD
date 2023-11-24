import torch
import numpy as np
import os
import math
import random
from collections import Counter
import scipy.io
import pickle
from scipy.stats import zscore


# har_modality = ['acce','gryo']
# acce_index = 0-62
# gyro_index = 63-77
# y_index = 78
# [['acce','gyro'],['acce'],['gyro']]


from collections import Counter


def get_opp_statistics(path):
    mat = scipy.io.loadmat(path)
    mat['x_train_acce'] = mat['x_train_acce'][:,39:]
    mat['x_test_acce'] = mat['x_test_acce'][:,39:]

    mat['x_train_acce'] = zscore(mat['x_train_acce'])
    mat['x_train_gyro'] = zscore(mat['x_train_gyro'])
    mat['x_test_acce'] = zscore(mat['x_test_acce'])
    mat['x_test_gyro'] = zscore(mat['x_test_gyro'])
    acce_min = min([mat['x_train_acce'].min(),mat['x_test_acce'].min()])
    acce_max = max([mat['x_train_acce'].max(),mat['x_test_acce'].max()])
    gyro_min = min([mat['x_train_gyro'].min(),mat['x_test_gyro'].min()])
    gyro_max = max([mat['x_train_gyro'].max(),mat['x_test_gyro'].max()])
    return acce_min, acce_max, gyro_min, gyro_max


def gen_opp_client_data(path,clients_config,overlap,T):
    statistics = get_opp_statistics(path)

    test_fraction = 0.2

    mat = scipy.io.loadmat(path)
    mat['x_train_acce'] = mat['x_train_acce'][:,39:]
    mat['x_test_acce'] = mat['x_test_acce'][:,39:]

    mat['x_train_acce'] = zscore(mat['x_train_acce'])
    mat['x_train_gyro'] = zscore(mat['x_train_gyro'])
    mat['x_test_acce'] = zscore(mat['x_test_acce'])
    mat['x_test_gyro'] = zscore(mat['x_test_gyro'])

    y = np.concatenate([mat['y_train'].squeeze(),mat['y_test'].squeeze()], axis=0)

    all_data = np.concatenate([np.concatenate([mat['x_train_acce'],mat['x_train_gyro']],axis=1),np.concatenate([mat['x_test_acce'],mat['x_test_gyro']],axis=1)],axis=0)
    all_y = y
    sequence_length = len(all_data)
    step = math.floor(T - overlap*T)

    all_samples = []
    all_labels = []
    start = 0
    end = T
    while end <= sequence_length:
        if np.bincount(all_y[start:end]).argmax() == 0:
            start+=step
            end+=step
            continue
        all_samples.append(all_data[start:end])
        all_labels.append(np.bincount(all_y[start:end]).argmax()-1)
        start+=step
        end+=step
    num_sample = len(all_samples)

    n_client = len(clients_config)
    sample_per_client = math.floor(num_sample / n_client)
    client_start = 0
    id = 0

    for config in clients_config:
        client_samples = np.stack(all_samples[client_start:client_start+sample_per_client], axis=0)
        client_labels = np.array(all_labels[client_start:client_start+sample_per_client])

        client_shuffle_idx = np.random.permutation(len(client_samples))
        client_samples,client_labels = client_samples[client_shuffle_idx], client_labels[client_shuffle_idx]

        if 'acce' not in config:
            client_samples[:,:,0:63] = 0
        if 'gyro' not in config:
            client_samples[:,:,63:78] = 0

        train_num = math.floor(sample_per_client * (1-test_fraction))

        client_train_samples = client_samples[0:train_num]
        client_train_labels = client_labels[0:train_num]
        client_test_samples = client_samples[train_num:]
        client_test_labels = client_labels[train_num:]

        print('client{0} class distribution:'.format(id))
        print(np.bincount(client_train_labels))
        print(np.bincount(client_test_labels))
        client_data = {'x_train':client_train_samples,'y_train':client_train_labels,'x_test':client_test_samples,'y_test':client_test_labels,'info':config,'id':id,'statistics':statistics}

        with open('client_'+str(id)+'_data', 'wb') as client_data_file:
            pickle.dump(client_data, client_data_file)

        id += 1
        client_start += sample_per_client









if __name__ == '__main__':
    gen_opp_client_data("../opp.mat",[['acce','gyro'],['acce','gyro'],['acce'],['acce']],0.75,128)

    #gen_har_client_data("../har.mat",[['acce']]*30,0.75,128)

    '''
    with open('client_2_data', 'rb') as data_file:
        data = pickle.load(data_file)
    print(data)

    print(len(data['x_train']))
    print(len(data['x_test']))
    assert len(data['x_train']) == len(data['y_train'])
    assert len(data['x_test']) == len(data['y_test'])
    mat = scipy.io.loadmat("../har/har.mat")
    print(mat['acce_train'].shape)
    print(data['y_train'][1:100])
    '''


