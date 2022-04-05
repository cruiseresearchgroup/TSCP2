import os

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tqdm import tqdm
import tensorflow as tf
from tcn import TCN
import numpy as np

import losses as ls
from utils.usc_ds_helper import ts_samples

#@tf.function
def train_step(xis, xjs, amodel, optimizer, criterion, temperature, sfn, lfn, beta, tau):
    # print("---------",xis.shape)
    with tf.GradientTape() as tape:
        zis = amodel(xis)
        zjs = amodel(xjs)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        #loss, mean_sim = ls.dcl_loss_fn(zis, zjs, temperature, lfn)
        loss, mean_sim, neg_sim = ls.loss_fn(zis, zjs, similarity = sfn, loss_fn = lfn, temperature=temperature, tau = tau, beta = beta, elimination_th = 0, attraction = False)

    gradients = tape.gradient(loss, amodel.trainable_variables)
    optimizer.apply_gradients(zip(gradients, amodel.trainable_variables))

    return loss, mean_sim, neg_sim


def train_prep(model, dataset, outpath, optimizer, criterion, train_name, win, temperature=0.1, epochs=100, sfn="cosine", lfn='nce', beta=0.1, tau=0.1):
    beta_curr = beta
    epoch_wise_loss = []
    epoch_wise_sim = []
    epoch_wise_neg = []
    end_condition = 0
    for epoch in tqdm(range(epochs)):
        counter = 0
        step_wise_loss = []
        step_wise_sim = []
        step_wise_neg = []
        for mbatch in dataset:
            counter += 1
            a, b, lbl = ts_samples(mbatch, win)

            # a = data_augmentation(mbatch)
            # b = data_augmentation(mbatch)

            loss, sim, neg = train_step(tf.expand_dims(a, axis=2), tf.expand_dims(b, axis=2), model, optimizer, criterion,
                                   temperature, sfn, lfn, beta=beta_curr, tau=tau)
            step_wise_loss.append(loss)
            step_wise_sim.append(sim)
            step_wise_neg.append(neg)

        epoch_wise_loss.append(np.mean(step_wise_loss))
        epoch_wise_sim.append(np.mean(step_wise_sim))
        epoch_wise_neg.append(np.mean(step_wise_neg))
        # wandb.log({"nt_INFONCEloss": np.mean(step_wise_loss)})
        # wandb.log({"nt_sim": np.mean(step_wise_sim)})
        #if epoch % (np.floor(epoch / 10)) == 0:
        #    beta_curr = beta_curr - (beta/10)
        if epoch % 1 == 0:
            result = "epoch: {} (step:{}) -loss: {:.3f} - avg rep sim : {:.3f} - avg rep neg : {:.3f}\n".format(epoch + 1,
                                                                                        counter,
                                                                                        np.mean(step_wise_loss),
                                                                                        np.mean(step_wise_sim),
                                                                                        np.mean(step_wise_neg))
            with open(os.path.join(outpath, train_name + ".txt"), "a") as myfile:
                myfile.write("{:.4f},{:.4f},{:.4f}\n".format(np.mean(step_wise_loss),
                                                             np.mean(step_wise_sim),
                                                             np.mean(step_wise_neg)))
                myfile.close()
            print(result)
        if epoch > 5:
            if np.abs(epoch_wise_loss[-1] - epoch_wise_loss[-2]) < 0.0001 or epoch_wise_loss[-2] < epoch_wise_loss[-1]:
                end_condition += 1
            else:
                end_condition = 0
            if end_condition == 4:
                return epoch_wise_loss, epoch_wise_sim, epoch_wise_neg, model

    return epoch_wise_loss, epoch_wise_sim, epoch_wise_neg, model


# Dilated-TCN
def get_TCN_encoder(f_shape, n_steps, code_size, nb_filters=64, kernel_size=4, nb_stacks=2, dilations=[1, 2, 4, 8],
                    padding='causal',
                    use_skip_connections=True, dropout_rate=0, activation='relu'):
    inputs = Input(f_shape)
    xrep = TCN(nb_filters, kernel_size, nb_stacks, dilations, padding,
               use_skip_connections, dropout_rate, return_sequences=True, activation='relu',
               kernel_initializer='random_normal', use_batch_norm=True)(inputs)
    # base_model.trainable = True

    xrep = Flatten()(xrep)
    xrep = Dense(2 * n_steps)(xrep)
    xrep = Activation("relu")(xrep)
    xrep = Dense(n_steps)(xrep)
    xrep = Activation("relu")(xrep)
    xrep = Dense(code_size)(xrep)
    # xrep_dtcn.append(xrep)

    # encoded = concatenate(xrep_dtcn, axis=-1)
    # encoded = BatchNormalization()(xrep)
    encoder = Model(inputs, xrep)

    return encoder

