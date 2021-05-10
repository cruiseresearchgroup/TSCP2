import numpy as np
import tensorflow as tf
from .hasc_helper import load_hasc_ds
from .usc_ds_helper import load_usc_ds

def load_dataset(path, ds_name, win, bs, mode="train"):
    if ds_name == 'HASC':
        trainx, trainlbl = load_hasc_ds(path, window = 2 * win, mode=mode)
    elif ds_name == "USC":
        trainx, trainlbl = load_usc_ds(path, window=2 * win, mode=mode)
    else:
        raise ValueError("Undefined Dataset.")

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
