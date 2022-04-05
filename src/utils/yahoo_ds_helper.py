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


def load_yahoo_ds(path, window, mode='train'):
    save_path = os.path.join("../data/yahoo")

    X, lbl = extract_windows(path, window, 5, save_path, save=False)

    if mode == "all":
        return X, lbl

    train_size = floor(0.6 * X.shape[0])
    if mode =="train":
        trainx = X[0:train_size,:]
        trainlbl =lbl[0:train_size]
        idx = np.arange(trainx.shape[0])
        np.random.shuffle(idx)
        trainx = trainx[idx,]
        trainlbl = trainlbl[idx]
        print('train samples : ', train_size)
        return trainx, trainlbl

    else:
        testx = X[train_size:,]
        testlbl = lbl[train_size:]
        print('test shape {} and number of change points {} '.format(testx.shape, len(np.where(testlbl>0)[0])))

        return testx, testlbl


def remove_anomalies(TS, anomalies):
    index = np.where(anomalies == 1)[0]
    # print('anomaly :',index)
    for i in index:

        if i > 0 and i + 1 < TS.shape[0]:
            TS[i] = (TS[i - 1] + TS[i + 1]) / 2
        elif i > 0:
            TS[i] = TS[i - 1]
        if i == 0:
            TS[i] = TS[i + 1]
    return TS


def extract_windows(path, window_size, step, save_path, save=False):
    files = os.scandir(path)
    window_size
    windows = []
    lbl = []
    first = True
    num_cp = 0
    for f in files:

        if f.name != 'A4Benchmark_all.csv':
            data = pd.read_csv(f)
            cp = data['changepoint']
            ts = remove_anomalies(data['value'].values, data['anomaly'])
            ts = (ts - np.min(ts)) / (np.max(ts) - np.min(ts))
            # ts = ts.values
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
        first = False

    print("number of samples : {} /  number of samples with change point : {}".format(len(windows), num_cp))
    windows = np.array(windows)
    if save:
        outfile = open(os.path.join(save_path, 'norm_A4_all_window' + str(window_size) + ".csv"), 'w', newline='')
        writer = csv.writer(outfile)
        writer.writerows(windows)
        outfile.close()

        outfile = open(os.path.join(save_path, 'norm_A4_all_window_label' + str(window_size) + ".csv"), 'w', newline='')
        writer = csv.writer(outfile)
        writer.writerows(np.array(lbl))
        outfile.close()
    return windows, np.array(lbl)