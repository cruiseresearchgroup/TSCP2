import csv
import os
import numpy as np
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.metrics import mean_squared_error

from utils.FastDTW import fastdtw


def getMSE(Yt, y_pred, sq = False):
    y_hat = y_pred.reshape((Yt.shape[0], Yt.shape[1], Yt.shape[2]))
    print('y_hat : ', y_hat.shape)
    loss = np.zeros((Yt.shape[0], Yt.shape[1]))
    err_total = np.zeros((Yt.shape[0]))
    for i in range(0, Yt.shape[0]):
        a = Yt[i, :, :].T
        b = y_hat[i, :, :].T
        err = mean_squared_error(a, b, multioutput='raw_values')
        loss[i, :] = err
        err_total[i] = mean_squared_error(a, b)
    return loss, err_total

def getPE(Yt, y_pred, sq = False):
    y_hat = y_pred.reshape((Yt.shape[0], Yt.shape[1], Yt.shape[2]))
    print('y_hat : ', y_hat.shape)
    loss = np.zeros((Yt.shape[0], Yt.shape[1]))
    err_total = np.zeros((Yt.shape[0]))
    for i in range(0, Yt.shape[0]):
        a = Yt[i, :, :].T
        b = y_hat[i, :, :].T
        err = (a-b)/b
        loss[i, :] = err
        err_total[i] = np.mean(err)
    return loss, err_total

def getMSPE(Yt, y_pred, sq = False):
    y_hat = y_pred.reshape((Yt.shape[0], Yt.shape[1], Yt.shape[2]))
    loss = np.zeros((Yt.shape[0], Yt.shape[1]))
    err_total = np.zeros((Yt.shape[0]))
    for i in range(0, Yt.shape[0]):
        a = Yt[i, :, :]
        b = y_hat[i, :, :]
        mse = np.mean(np.abs(a-b)/(np.abs(np.max(b)-np.min(b))), axis=1) #* (np.abs(np.abs(np.max(b)-np.min(b))-np.abs(np.max(a)-np.min(a))))
        loss[i,:]=mse
        err_total[i] = np.mean(mse)
    return loss, err_total

def remove_outliers(data, m=2):
    for i in range(0, data.shape[1]):
        outliers = abs(data[:,i] - np.mean(data[:,i])) >= m * np.std(data[:,i])
        print('Min({}), Max({}), median :{},  mean : {} and std {} '.format(np.min(data[:,i]),np.max(data[:,i]),np.median(data[:,i]),np.mean(data[:,i]),np.std(data[:,i])))
        #data[outliers,i]= m * np.std(data[:,i])

    #print(outlier)
    return data

def plot_predict(y, p, model):
    plt.plot(y, label='real data')
    plt.plot(p, label='prediction')
    plt.legend()
    plt.title(model + " (loss" + str(mean_squared_error(y, p)) + ")")

def getDTW(Yt, y_pred):
    y_hat = y_pred.reshape((Yt.shape[0], Yt.shape[1], Yt.shape[2]))
    print('y_hat : ', y_hat.shape)
    dtw = np.zeros((Yt.shape[0], Yt.shape[1]))
    #dtw_total = np.zeros((Yt.shape[0]))
    for i in range(0, Yt.shape[0]):
        for j in range(0, Yt.shape[1]):
            dtw[i, j],_  = fastdtw(Yt[i, j, :], y_hat[i, j, :])
    dtw_total = np.mean(dtw, axis=1)
    print('dtw : {} , dta_total : {}'.format(dtw.shape,dtw_total.shape))
    return dtw, dtw_total

def dtw_boundaries(dtw_tot, order, cp_size,PATH, MODEL_NAME, option_str,mode):
    errors = np.gradient(np.array(dtw_tot)).tolist()
    predBoundaries = signal.argrelextrema(np.asarray(errors), np.greater, order=order)[0].tolist()
    predBoundaries.append(cp_size - 1)
    writer = csv.writer(
        open(os.path.join(PATH, MODEL_NAME + option_str + "-"+mode+"-boundaries-order"+ str(order)+".csv"), "w", newline=''))
    writer.writerows(np.array([[x] for x in predBoundaries]))


def rep_visu(x_test,win,history, future, lbl):
    N = 20
    id = np.random.randint(0, history.shape[0], N)

    plt.figure(figsize=(15, 40))
    for i in range(0, N):
        plt.subplot(N, 2, 1 + i * 2)
        plt.title('actual ts' + str(lbl[id[i]]))
        plt.plot(x_test[id[i], 0:win], color='k')
        plt.plot(x_test[id[i], win:], color='b')
        plt.legend(['history', 'future'])

        plt.subplot(N, 2, 2 + i * 2)
        plt.title('reps')
        plt.plot(history[id[i]], color='k')
        plt.plot(future[id[i]], color='b')
        # plt.legend(['history','future'])
    plt.show()