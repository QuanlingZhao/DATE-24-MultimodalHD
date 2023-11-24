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
# acce_index = 0-5
# gyro_index = 6-8
# y_index = 9
# [['acce','gyro'],['acce'],['gyro']]


from collections import Counter


def get_har_statistics(path):
    mat = scipy.io.loadmat(path)
    mat['acce_train'] = zscore(mat['acce_train'])
    mat['gyro_train'] = zscore(mat['gyro_train'])
    mat['acce_test'] = zscore(mat['acce_test'])
    mat['gyro_test'] = zscore(mat['gyro_test'])
    acce_min = min([mat['acce_train'].min(),mat['acce_test'].min()])
    acce_max = max([mat['acce_train'].max(),mat['acce_test'].max()])
    gyro_min = min([mat['gyro_train'].min(),mat['gyro_test'].min()])
    gyro_max = max([mat['gyro_train'].max(),mat['gyro_test'].max()])
    return acce_min, acce_max, gyro_min, gyro_max


def gen_har_client_data(path,clients_config,overlap,T):
    statistics = get_har_statistics(path)

    test_fraction = 0.2

    mat = scipy.io.loadmat(path)

    mat['acce_train'] = zscore(mat['acce_train'])
    mat['gyro_train'] = zscore(mat['gyro_train'])
    mat['acce_test'] = zscore(mat['acce_test'])
    mat['gyro_test'] = zscore(mat['gyro_test'])

    acce = np.concatenate([mat['acce_train'],mat['acce_test']], axis=0)
    gyro = np.concatenate([mat['gyro_train'],mat['gyro_test']], axis=0)
    y = np.concatenate([mat['y_train'].squeeze(),mat['y_test'].squeeze()], axis=0)

    all_data = np.concatenate([mat['acce_train'],mat['gyro_train']],axis=1)
    all_y = y
    sequence_length = len(all_data)
    step = math.floor(T - overlap*T)


    all_samples = []
    all_labels = []
    start = 0
    end = T
    while end <= sequence_length:
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
            client_samples[:,:,0:6] = 0
        if 'gyro' not in config:
            client_samples[:,:,6:9] = 0

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
    gen_har_client_data("../har.mat",[['acce','gyro'],['acce'],['gyro']]*10,0.75,128)

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


