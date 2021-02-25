import numpy as np
from sklearn.metrics import confusion_matrix,f1_score
from matplotlib import pyplot as plt
from losses import _cosine_simililarity_dim1


def estimate_CPs(sim, gt, name, train_name, metric='cosine', threshold=0.5):
    #if metric == "cosine":
    #    sim = _cosine_simililarity_dim1(h, f)

    est_cp = np.zeros(sim.shape[0])
    est_cp[np.where(sim < threshold)[0]] = 1
    tn, fp, fn, tp = confusion_matrix(gt, est_cp).ravel()
    f1 = f1_score(gt, est_cp)

    ## gt==1
    gt_id = np.where(gt == 1)[0]
    """
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 1, 1)
    for i in gt_id:
        plt.axvline(x=i, ymin=0, ymax=1, color='k')
    plt.subplot(2, 1, 2)
    for i in np.where(est_cp == 1)[0]:
        plt.axvline(x=i, ymin=0, ymax=1, color='r')
    plt.savefig(name+".png")
    plt.savefig(name + ".pdf")
    """
    print("tn {}, fp {}, fn {}, tp {} ----- f1-score {}".format(tn, fp, fn, tp, f1))

    ## continuous series
    i = 1
    pos, seq_tp, seq_fn, seq_fp = 0, 0, 0, 0

    while i < gt.shape[0]:
        if gt[i] == 1:
            pos += 1
            j = i
            while gt[i] == 1:
                i += 1

            if np.sum(est_cp[j:i]) > 0:
                seq_tp += 1
                est_cp[j:i] = 0
            else:
                seq_fn += 1

        i += 1

    seq_fp = np.where(np.diff(est_cp) == 1)[0].shape[0]
    seq_f1 = (2 * seq_tp) / (2 * seq_tp + seq_fn + seq_fp)

    print("SEQ : Pos {}, fp {}, fn {}, tp {} ----- f1-score {}".format(pos, seq_fp, seq_fn, seq_tp, seq_f1))
    result = "tn, {}, fp, {}, fn, {}, tp, {}, f1-score, {}, Pos, {}, seqfp, {}, seqfn, {}, seqtp, {}, seqf1, {}\n".format(tn, fp, fn, tp, f1, pos, seq_fp, seq_fn, seq_tp, seq_f1)
    return result