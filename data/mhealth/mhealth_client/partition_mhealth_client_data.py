import torch
import numpy as np
import os
import math
import random
from collections import Counter
import scipy.io
import pickle
from scipy.stats import zscore


# har_modality = ['acce','gryo','mage']
# acce_index = 0-8
# gyro_index = 9-14
# y_index = 15-20
# [['acce','gyro','mage'],['acce','gyro'],['acce','mage'],['gyro','mage']]

def get_mhealth_statistics(path):
    mat = scipy.io.loadmat(path)
    acce = []
    gyro = []
    mage = []
    for s in range(1,11):
        acce.append(mat['s{0}_acce'.format(str(s))])
        gyro.append(mat['s{0}_gyro'.format(str(s))])
        mage.append(mat['s{0}_mage'.format(str(s))])
    acce = zscore(np.concatenate(acce, axis=0))
    gyro = zscore(np.concatenate(gyro, axis=0))
    mage = zscore(np.concatenate(mage, axis=0))

    acce_max = acce.max()
    acce_min = acce.min()
    gyro_max = gyro.max()
    gyro_min = gyro.min()
    mage_max = mage.max()
    mage_min = mage.min()

    return acce_min, acce_max, gyro_min, gyro_max, mage_min, mage_max



def gen_health_client_data(path,clients_config,overlap,T):
    statistics = get_mhealth_statistics(path)

    test_fraction = 0.2
    mat = scipy.io.loadmat(path)

    acce = []
    gyro = []
    mage = []
    y = []
    for s in range(1,11):
        acce.append(mat['s{0}_acce'.format(str(s))])
        gyro.append(mat['s{0}_gyro'.format(str(s))])
        mage.append(mat['s{0}_mage'.format(str(s))])
        y.append(mat['s{0}_y'.format(str(s))].squeeze())
    acce = zscore(np.concatenate(acce, axis=0))
    gyro = zscore(np.concatenate(gyro, axis=0))
    mage = zscore(np.concatenate(mage, axis=0))
    y = np.concatenate(y, axis=0)

    all_data = np.concatenate([acce,gyro,mage],axis=1)
    all_y = y
    sequence_length = len(all_data)
    step = math.floor(T - overlap*T)

    all_samples = []
    all_labels = []
    start = 0
    end = T
    while end <= sequence_length:
        all_samples.append(all_data[start:end])
        all_labels.append(np.bincount(all_y[start:end]).argmax())
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
            client_samples[:,:,0:9] = 0
        if 'gyro' not in config:
            client_samples[:,:,9:15] = 0
        if 'mage' not in config:
            client_samples[:,:,15:21] = 0

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
    gen_health_client_data('../mhealth.mat',[['acce','gyro','mage'],['acce','gyro','mage'],['acce','gyro','mage'],['acce','gyro'],['acce','gyro'],
        ['acce','gyro'],['acce','mage'],['acce','mage'],['acce','mage'],['acce','mage']],0.75,128)

    #gen_health_client_data('../mhealth.mat',[['acce'],['acce'],['acce'],['acce'],['acce'],
    #    ['acce'],['acce'],['acce'],['acce'],['acce']],0.75,128)

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
    print(np.unique(data['y_train']))
    '''










