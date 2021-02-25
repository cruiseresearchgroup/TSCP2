# import the necessary packages
import numpy as np
import random
import os
from pandas import read_csv

from utils.DataHelper import load_dataset, load_WISDM_dataset


def training_generator(data_path, dataset, steps_in, steps_out, batch_size, positive_samples=1, mode="train", aug=False):
    # bs : Batch size
    # lb :
    # loop indefinitely
    if positive_samples > batch_size // 2 :
        positive_samples = batch_size //2

    # Load Training Data
    data_x, data_y, data_lbl = load_dataset(data_path, dataset, mode, step_in=steps_in, step_out=steps_out)

    #data_x, data_y, data_lbl = data_x[:1000, ], data_y[:1000, ], data_lbl[:1000, ]

    # shuffle the instances
    idx = np.arange(data_x.shape[0])
    np.random.shuffle(idx)
    data_x = data_x[idx,:,:]
    data_y = data_y[idx, :, :]
    data_lbl = data_lbl[idx]

    curr_id = 0
    while True:
        # initialize our batches of images and labels
        #pairs = np.empty(batch_size, dtype=object)
        x = np.empty((batch_size,data_x.shape[1],data_x.shape[2]), dtype=float)
        y = np.empty((batch_size,data_y.shape[1],data_y.shape[2]), dtype=float)
        labels = []

        ind = 0
        # keep looping until we reach our batch size
        p_sample_n = 0
        i = 0
        while i < batch_size:
            # attempt to read the next subsequence
            if curr_id >= data_x.shape[0]:
                curr_id = 0
            if p_sample_n < positive_samples:
                # positive pair
                x[i,], y[i,]= [data_x[curr_id,:], data_y[curr_id, :]]
                labels += [1]
                p_sample_n +=1
                curr_id += 1

            else:
                # add negative pairs
                cid = random.choice(np.arange(curr_id-positive_samples,curr_id))
                neg_id = random.choice(np.where(data_lbl!=data_lbl[cid])[0])
                x[i,],y[i,] = data_x[cid, :], data_y[neg_id, :]
                labels +=[0]
            #print(curr_id , "-",i)
            i += 1

        # yield the batch to the calling function
        yield ([np.array(x), np.array(y)], [np.array(labels)])


def wisdm_training_generator(data_path, steps_in, steps_out, batch_size, positive_samples=1, mode="train", aug=False, part=1):
    # bs : Batch size
    # lb :
    # loop indefinitely
    if positive_samples > batch_size // 2 :
        positive_samples = batch_size // 2

    # Load Training Data
    data_x, data_y, data_lbl = load_WISDM_dataset(data_path, mode, step_in=steps_in, step_out=steps_out, overlap=0.9, part=part)
    curr_id = 0
    while True:

        x = np.empty((batch_size,data_x.shape[1],data_x.shape[2]), dtype=float)
        y = np.empty((batch_size,data_y.shape[1],data_y.shape[2]), dtype=float)
        labels = []
        if curr_id >= data_x.shape[0]:
            curr_id=0

        # keep looping until we reach our batch size
        p_sample_n = 0
        i = 0
        while i < batch_size:
            # attempt to read the next subsequence
            if p_sample_n < positive_samples:
                # positive pair
                x[i,], y[i,]= [data_x[curr_id,:], data_y[curr_id, :]]
                labels += [1]
                p_sample_n +=1
                i += 1
                curr_id += 1
                #if aug:
                    # add augmented positive pair
                    #pairs += [augmented_pair(data_x[curr_id, :], data_y[curr_id, :])]
                    #labels += [1]
                    #p_sample_n += 1

            else:
                # add negative pairs
                cid = random.choice(np.arange(curr_id-positive_samples,curr_id))
                neg_id = random.choice(np.where(data_lbl!=data_lbl[cid])[0])
                x[i,],y[i,] = data_x[cid, :], data_y[neg_id, :]
                labels +=[0]
                i += 1

        # yield the batch to the calling function
        #yield ([np.array(x),np.array(y)], [np.array(labels),np.array(y).reshape((y.shape[0],-1))])
        yield ([np.array(x), np.array(y)], [np.array(labels)])



def semi_hard_training_generator(data_path, dataset, steps_in, steps_out, batch_size, mode="train", aug=False, part=1):

    # Load Training Data
    if dataset == 'WISDM':
        data_x, data_y, data_lbl = load_WISDM_dataset(data_path, mode, step_in=steps_in, step_out=steps_out,                                                    overlap=0.9, part=part)
    else:
        data_x, data_y, data_lbl = load_dataset(data_path, dataset, mode, step_in=steps_in, step_out=steps_out)

    # data_x, data_y, data_lbl = data_x[:1000, ], data_y[:1000, ], data_lbl[:1000, ]
    # shuffle the instances
    if mode == "test":
        batch_size=1
    else:
        idx = np.arange(data_x.shape[0])
        np.random.shuffle(idx)
        data_x = data_x[idx, :, :]
        data_y = data_y[idx, :, :]

    curr_id = 0
    while True:
        # initialize our batches of images and labels
        x = np.empty((batch_size, data_x.shape[1], data_x.shape[2]), dtype=float)
        y = np.empty((batch_size, data_y.shape[1], data_y.shape[2]), dtype=float)


        if curr_id >= data_x.shape[0]-1:
            curr_id = 0
        # keep looping until we reach our batch size
        i = 0
        while i < batch_size:
            # attempt to read the next subsequence
            x[i,] = data_x[curr_id, ]
            y[i,] = data_y[curr_id, ]


            i += 1
            curr_id += 1
            if curr_id == data_x.shape[0]:
                curr_id = 0
            # if aug:

        #labels = np.identity(i)
        labels = np.ones(i)
        # yield the batch to the calling function
        yield ([np.array(x), np.array(y)], labels)  # x: (bs, dim, history_window) , y:(bs, dim, future_window), lbl:(bs,bs)
