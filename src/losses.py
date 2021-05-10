from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

cosine_sim_1d = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
cosine_sim_2d = tf.keras.losses.CosineSimilarity(axis=2, reduction=tf.keras.losses.Reduction.NONE)

def _cosine_simililarity_dim1(x, y):
    v = cosine_sim_1d(x, y)
    return -v

def _cosine_simililarity_dim2(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    v = cosine_sim_2d(tf.expand_dims(x, 1), tf.expand_dims(y, 0))
    return -v


def _dot_simililarity_dim1(x, y):
    # x shape: (N, 1, C)
    # y shape: (N, C, 1)
    # v shape: (N, 1, 1)
    v = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))
    return v

def _dot_simililarity_dim2(x, y):
    v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)
    # x shape: (N, 1, C)
    # y shape: (1, C, 2N)
    # v shape: (N, 2N)
    return v


def _euclidean_similarity_dim1(x, y):
    # x shape: (N, 1, C)
    # y shape: (N, 1, C)
    # v shape: (N, 1, 1)
    d = tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=-1))
    s = 1 / ( 1 + d )
    return s

def _euclidean_similarity_dim2(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, N, C)
    # v shape: (N, 1, 1)
    x1 = tf.expand_dims(x,1)
    y1 = tf.expand_dims(y,0)
    d = tf.sqrt(tf.reduce_sum(tf.square(x1 - y1), axis=2))
    s = 1 / (1 + d)
    return s


def _edit_similarity_dim1(x, y):
    # x shape: (N, 1, C)
    # y shape: (N, 1, C)
    # v shape: (N, 1, 1)
    d = tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=-1))
    s = 1 / ( 1 + d )
    return s

def _edit_similarity_dim2(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, N, C)
    # v shape: (N, 1, 1)
    x1 = tf.expand_dims(x,1)
    y1 = tf.expand_dims(y,0)
    d = tf.sqrt(tf.reduce_sum(tf.square(x1 - y1), axis=2))
    s = 1 / (1 + d)
    return s

def loss_fn(history, future, similarity,  loss_fn = "nce", temperature=0.1, tau=0.1, beta=0.1,  elimination_th = 0, elimination_topk = 0.1, attraction = False):


    if loss_fn == "nce":
        loss, pos, neg = nce_loss_fn(history, future, similarity, temperature)
    elif loss_fn == "dcl":
        loss, pos, neg = dcl_loss_fn(history, future, similarity, temperature, debiased = True, tau_plus = tau)
    elif loss_fn == "fc":
        loss, pos, neg = fc_loss_fn(history, future, similarity, temperature , elimination_th = elimination_th, elimination_topk = beta, attraction = attraction)
    elif loss_fn == "harddcl":
        loss, pos, neg = hard_loss_fn(history, future, similarity, temperature, beta=beta, debiased=True, tau_plus=tau)

    return loss, pos, neg

def nce_loss_fn(history, future, similarity, temperature='0.1'):
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)

    #pos_sim = ls._cosine_simililarity_dim1(history, future) / temperature
    #neg_sim = ls._cosine_simililarity_dim2(history, future) / temperature
    N = history.shape[0]
    sim = similarity(history, future)
    pos_sim = K.exp(tf.linalg.tensor_diag_part(sim)/temperature)

    tri_mask = np.ones(N ** 2, dtype=np.bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False
    neg = tf.reshape(tf.boolean_mask(sim, tri_mask), [N, N - 1])
    all_sim = K.exp(sim/temperature)

    logits = tf.divide(K.sum(pos_sim), K.sum(all_sim, axis=1))

    lbl = np.ones(history.shape[0])
    # categorical cross entropy
    loss = criterion(y_pred = logits, y_true = lbl)
    # loss = K.sum(logits)
    # divide by the size of batch
    #loss = loss / lbl.shape[0]
    # similarity of positive pairs (only for debug)
    mean_sim = K.mean(tf.linalg.tensor_diag_part(sim))
    mean_neg = K.mean(neg)
    return loss, mean_sim, mean_neg


def dcl_loss_fn(history, future, similarity, temperature='0.1', debiased=True, tau_plus=0.1):
    # from Debiased Contrastive Learning paper: https://github.com/chingyaoc/DCL/
    # pos: exponential for positive example
    # neg: sum of exponentials for negative examples
    # N : number of negative examples
    # t : temperature scaling
    # tau_plus : class probability

    N = history.shape[0]
    sim = similarity(history, future)
    pos_sim = K.exp(tf.linalg.tensor_diag_part(sim)/temperature)

    tri_mask = np.ones(N ** 2, dtype=np.bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False
    neg = tf.reshape(tf.boolean_mask(sim, tri_mask),[N,N-1])
    neg_sim = K.exp(neg/temperature)

    # estimator g()
    if debiased:
        N = N-1
        Ng = (-tau_plus * N * pos_sim + K.sum(neg_sim, axis=-1)) / (1 - tau_plus)
        # constrain (optional)
        Ng = tf.clip_by_value(Ng, clip_value_min=N * np.e ** (-1 / temperature), clip_value_max=tf.float32.max)
    else:
        Ng = K.sum(neg_sim, axis=-1)

    # contrastive loss
    loss = K.mean(- tf.math.log(pos_sim / (pos_sim + Ng)))

    # similarity of positive pairs (only for debug)
    mean_sim = K.mean(tf.linalg.tensor_diag_part(sim))
    mean_neg = K.mean(neg)
    return loss, mean_sim, mean_neg


def fc_loss_fn(history, future, similarity, temperature=0.1, elimination_th = 0, elimination_topk = 0.1, attraction = False):
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    N = history.shape[0]
    if elimination_topk > 0.5:
        elimination_topk = 0.5
    elimination_topk = np.math.ceil(elimination_topk * N)

    sim = similarity(history, future)/temperature

    pos_sim = K.exp(tf.linalg.tensor_diag_part(sim))

    tri_mask = np.ones(N ** 2, dtype=np.bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False
    neg_sim = tf.reshape(tf.boolean_mask(sim, tri_mask), [N, N - 1])

    #sorted_ind = tf.argsort(neg_sim, axis=1)
    sorted_sim = tf.sort(neg_sim, axis=1)
    if elimination_th > 0:
        # Threshold-base cancellation only --TODO
        threshold = tf.constant([elimination_th])
        mask =  tf.cast(tf.math.greater(threshold,sorted_sim),tf.float32)
        neg_count =  tf.reduce_sum(mask,axis=1)
        neg = tf.divide(tf.reduce_sum(sorted_sim * mask, axis=1),neg_count)
        neg_sim = tf.reduce_sum(K.exp(sorted_sim/temperature) * mask, axis=1)

    else:
        # Top-K cancellation only
        if elimination_topk == 0:
            elimination_topk = 1
        tri_mask = np.ones(N * (N - 1), dtype=np.bool).reshape(N, N - 1)
        tri_mask[:, -elimination_topk:] = False
        neg = tf.reshape(tf.boolean_mask(sorted_sim, tri_mask), [N, N - elimination_topk - 1])
        neg_sim = K.sum(K.exp(neg), axis=1)

    #logits = tf.divide(K.sum(pos_sim, axis=-1), pos_sim+neg_sim)
    #lbl = np.ones(N)
    # categorical cross entropy
    #loss = criterion(y_pred = logits, y_true = lbl)
    loss = K.mean(- tf.math.log(pos_sim / (pos_sim + neg_sim)))

    # divide by the size of batch
    #loss = loss / N
    # similarity of positive pairs (only for debug)
    mean_sim = K.mean(tf.linalg.tensor_diag_part(sim)) * temperature
    mean_neg = K.mean(neg) * temperature
    return loss, mean_sim, mean_neg

def hard_loss_fn(history, future, similarity, temperature, beta = 0, debiased = True, tau_plus = 0.1):
    # from ICLR2021 paper: Contrastive LEarning with Hard Negative Samples https://www.groundai.com/project/contrastive-learning-with-hard-negative-samples
    # pos: exponential for positive example
    # neg: sum of exponentials for negative examples
    # N : number of negative examples
    # t : temperature scaling
    # tau_plus : class probability
    #
    # reweight = (beta * neg) / neg.mean()
    # Neg = max((-N * tau_plus * pos + reweight * neg).sum() / (1 - tau_plus), e ** (-1 / t))
    # hard_loss = -log(pos.sum() / (pos.sum() + Neg))

    N = history.shape[0]

    sim = similarity(history, future)
    pos_sim = K.exp(tf.linalg.tensor_diag_part(sim)/temperature)

    tri_mask = np.ones(N ** 2, dtype=np.bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False
    neg = tf.reshape(tf.boolean_mask(sim, tri_mask),[N,N-1])
    neg_sim = K.exp(neg/temperature)

    reweight = (beta * neg_sim) / tf.reshape(tf.reduce_mean(neg_sim, axis=1), [-1, 1])
    if beta == 0:
        reweight = 1
    # estimator g()
    if debiased:
        N = N-1
        #(-N*tau_plus*pos + reweight*neg).sum() / (1-tau_plus)

        Ng = (-tau_plus * N * pos_sim + tf.reduce_sum(reweight * neg_sim, axis=-1)) / (1 - tau_plus)
        # constrain (optional)
        Ng = tf.clip_by_value(Ng, clip_value_min=np.e ** (-1 / temperature), clip_value_max=tf.float32.max)
    else:
        Ng = K.sum(neg_sim, axis=-1)

    # contrastive loss
    #loss = K.mean(- tf.math.log(pos_sim / (pos_sim + Ng)))
    loss = K.mean(-tf.math.log(pos_sim / (pos_sim + Ng)))
    # similarity of positive pairs (only for debug)
    mean_sim = K.mean(tf.linalg.tensor_diag_part(sim))
    mean_neg = K.mean(neg)
    return loss, mean_sim, mean_neg


def _neural_warp_dim2():
    loss, mean_sim, mean_neg = 0, 0, 0
    return loss, mean_sim, mean_neg