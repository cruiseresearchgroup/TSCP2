import numpy as np
from pandas import read_csv
import csv
from numpy import array
import os
from math import floor
import tensorflow as tf
from scipy import stats
# split a multivariate sequence into samples
from sklearn.preprocessing import MinMaxScaler

from utils.hasc_helper import load_hasc_ds
from utils.usc_ds_helper import load_usc_ds
from utils.wsdm_ds_helper import load_wsdm_ds
from utils.yahoo_ds_helper import load_yahoo_ds

def load_dataset(path, ds_name, win, bs, mode="train"):
    if ds_name == 'HASC':
        trainx, trainlbl = load_hasc_ds(path, window = 2 * win, mode=mode)
    elif ds_name == "YAHOO":
        trainx, trainlbl = load_yahoo_ds(path, window=2 * win, mode=mode)
    elif ds_name == "USC":
        trainx, trainlbl = load_usc_ds(path, window=2 * win, mode=mode)
    elif ds_name == "WISDM":
        trainx, trainlbl = load_wsdm_ds(path, window=2 * win, mode=mode)

    trainlbl = trainlbl.reshape((trainlbl.shape[0], 1))
    print(trainx.shape, trainlbl.shape)
    dataset = np.concatenate((trainlbl, trainx), 1)

    print("dataset shape : ", dataset.shape)
    if mode == "test":
        return dataset
    # Create TensorFlow dataset
    train_ds = tf.data.Dataset.from_tensor_slices(dataset)
    train_ds = (train_ds.batch(bs, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
    return train_ds
