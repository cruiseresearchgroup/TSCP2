import tensorflow as tf
import numpy as np
#from augmentation.gaussian_filter import GaussianBlur

def get_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.zeros((batch_size, batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 1
        #negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)

def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)

