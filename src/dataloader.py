import os
import math
import random
import logging
import scipy.io
import numpy as np


LOADER_MODE = ['train', 'test', 'param']


def parse_txt(fname, precision=np.float32):
    if not os.path.isfile(fname):
        logging.error('cannot finde filepath: {0}'.format(fname))
        return np.asarray([])

    with open(fname, 'r') as f:
        content = f.readlines()

    data = []
    for c in content:
        data += [d.strip() for d in c.split('\t') if d not in ['', '\t', '\n']]

    data = np.asarray(data, dtype=precision)
    return data


def mat_data_loader(dpath, part_id):
    if not os.path.isdir(dpath):
        logging.error('cannot find dataset path: {0}'.format(dpath))
        return None

    if part_id == 1:
        logging.warning('not implementation mat_data_loader part_id=1 now.')
    elif part_id == 2:
        data = scipy.io.loadmat(os.path.join(dpath, 'PA-1-data-matlab/count_data.mat'))
        data['trainx'] = np.transpose(data['trainx'])
        data['testx'] = np.transpose(data['testx'])
        data['trainy'] = data['trainy'].flatten()
        data['testy'] = data['testy'].flatten()
        data['ym'] = data['ym'].flatten()
    return data


def txt_data_loader(dpath, part_id, mode='train'):
    if not os.path.isdir(dpath):
        logging.error('cannot find dataset path: {0}'.format(dpath))
        return None
    elif mode not in LOADER_MODE:
        logging.error('loader mode must in {0}'.format(LOADER_MODE))
        return None

    if part_id == 1:
        if mode == 'param':
            filep = os.path.join(dpath, 'PA-1-data-text/polydata_data_thtrue.txt')
            param = parse_txt(filep)
            return param
        else:
            suffix = 'samp' if mode == 'train' else 'poly'
            filex = os.path.join(dpath, 'PA-1-data-text/polydata_data_' + suffix + 'x.txt')
            filey = os.path.join(dpath, 'PA-1-data-text/polydata_data_' + suffix + 'y.txt')
            datax, datay = parse_txt(filex), parse_txt(filey)

            assert datax.shape == datay.shape, \
                'x: {0} and y: {1} need to have same shape'.format(datax.shape, datay.shape)
            return datax, datay

    elif part_id == 2:
        logging.warning('not implementation now.')

    else:
        logging.error('unknown task part1: {0}'.format(dpath))


def create_subset(datax, datay, precent=1):
    if precent == 1:
        return datax, datay 

    assert precent < 1 and precent > 0
    assert datax.shape[0] == datay.shape[0], \
        'x: {0} and y: {1} need to have same sample number'.format(datax.shape[0], datay.shape[0])

    total_size = len(datax)
    subset_size = math.floor(total_size * precent)
    assert subset_size > 0

    idxs = random.sample(range(0, total_size), subset_size)
    return np.copy(datax)[idxs], np.copy(datay)[idxs]
    

def add_outliers(algorithm, datax, datay, sigma, outliersx):
    if outliersx == []:
        return datax, datay

    outliersy = algorithm.predict(outliersx)
    noise = np.random.normal(0, sigma, len(outliersx))
    outliersy += noise

    datax = np.hstack((datax, outliersx))
    datay = np.hstack((datay, outliersy))

    return datax, datay, outliersy


def create_n_fold_dataset(datax, datay, n=1, shuffle=True):
    if n <= 1:
        return datax, datay

    assert isinstance(n, int)    
    assert datax.shape[0] == datay.shape[0], \
        'x: {0} and y: {1} need to have same sample number'.format(datax.shape[0], datay.shape[0])

    total_size = len(datax)

    fold_size = math.floor(total_size / n)
    assert fold_size > 0

    if shuffle:
        shuffle_idx = np.random.permutation(total_size)
        shufflex, shuffley = np.copy(datax)[shuffle_idx], np.copy(datay)[shuffle_idx] 
    else:
        shufflex, shuffley = np.copy(datax), np.copy(datay)
    n_dataset = []

    for k in range(0, n - 1):
        start = k * fold_size
        subx = shufflex[start : start + fold_size]
        suby = shuffley[start : start + fold_size]
        n_dataset.append([subx, suby])
    
    n_dataset.append([shufflex[(n-1) * fold_size: ], 
                      shuffley[(n-1) * fold_size: ]])

    return n_dataset
