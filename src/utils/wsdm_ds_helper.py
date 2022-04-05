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

def load_wsdm_ds(path, window, mode='train', part=1):
    save_path = os.path.join("../data/slidingwindow")
    """
    print(os.getcwd())
    xfile_name = os.path.join(save_path,'norm_A4_all_window_label' + str(window) + ".csv")
    lblfile_name = os.path.join(save_path, 'norm_A4_all_window_label' + str(window) + ".csv")

    if os.path.isfile(xfile_name) == True:
        X = read_csv(xfile_name, header=None).values
        lbl = read_csv(lblfile_name, header=None).values
    else:
    """
    X, lbl = extract_windows(path, window, mode, part=part)

    if mode == "all":
        return X, lbl


    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx,]
    lbl = lbl[idx]

    id = np.where(lbl==0)[0]
    num_neg = X.shape[0] - id.size
    num_train = id.size - num_neg
    trainx = X[id]
    trainlbl = lbl[id]
    print('pure samples : ', id.size)
    if mode == 'train':
        return trainx[:num_train, ], trainlbl[:num_train, ]

    else:
        neg_id = np.where((lbl > int(window*0.1)) & (lbl < int(window*0.9)))[0]
        # print('neg samples : ', neg_id.size)
        testx = np.vstack((X[neg_id], trainx[num_train:, ]))
        testlbl = np.vstack((lbl[neg_id], trainlbl[num_train:, ]))
        # print('neg shape : ', testx.shape)
        idx = np.arange(testx.shape[0])
        np.random.shuffle(idx)
        testx = testx[idx,]
        testlbl = testlbl[idx,]
        # print('2neg shape : ', testx.shape)
        return testx, testlbl


def extract_windows(path, window_size, mode="train", part=1):
    #files = os.scandir(path)
    window_size
    windows = []
    lbl = []
    first = True
    num_cp = 0
    X, C= load_WISDM_dataset(path, mode, win=window_size, overlap=0.9, part=part)
    #X = np.concatenate((X,Y), axis=-1)

    windows = X
    lbl = C
    """
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
        """
    return windows, np.array(lbl)