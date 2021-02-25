from math import floor

import numpy as np
from pandas import read_csv
import pandas as pd
import csv
import os


def ts_samples(mbatch, win):

    x = mbatch[:,1:win+1]
    y = mbatch[:,-win:]
    lbl = mbatch[:,0]
    return x, y, lbl


def load_hasc_ds(path, window, mode='train'):

    X, lbl = extract_windows(path, window, 5)

    if mode == "all":
        return X, lbl

    train_size = floor(0.8 * X.shape[0])
    if mode =="train":
        trainx = X[0:train_size]
        trainlbl =lbl[0:train_size]
        idx = np.arange(trainx.shape[0])
        np.random.shuffle(idx)
        trainx = trainx[idx,]
        trainlbl = trainlbl[idx]
        print('train samples : ', train_size)
        return trainx, trainlbl

    else:
        testx = X[train_size:]
        testlbl = lbl[train_size:]
        print('test shape {} and number of change points {} '.format(testx.shape, len(np.where(testlbl>0)[0])))

        return testx, testlbl



def extract_windows(path, window_size, step):
    files = os.scandir(path)
    window_size
    windows = []
    lbl = []
    first = True
    num_cp = 0
    for f in files:
        dataset = pd.read_csv(f).values
        x = dataset[:,1:]
        cp = dataset[:,0]

        ts = np.sqrt(np.power(x[:, 0], 2) + np.power(x[:, 1], 2) + np.power(x[:, 2], 2))
        for i in range(0, ts.shape[0] - window_size, step):
             windows.append(np.array(ts[i:i + window_size]))
             # print("TS",ts[i:i+window_size])
             is_cp = np.where(cp[i:i + window_size] == 1)[0]
             if is_cp.size == 0:
                is_cp = [0]
             else:
                num_cp += 1
             lbl.append(is_cp[0])

             # print(is_cp)

    print("number of samples : {} /  number of samples with change point : {}".format(len(windows), num_cp))
    windows = np.array(windows)

    return windows, np.array(lbl)