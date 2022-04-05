import sys
import numpy as np
import argparse
import os
import tensorflow as tf
import matplotlib.pyplot as plt
# dataset
import TSCP2 as cp2
import losses as ls
from utils.DataHelper import load_dataset
from utils.estimate_CPD import estimate_CPs

parser = argparse.ArgumentParser(description='interface of running experiments for TSCP2 baselines')
parser.add_argument('--datapath', type=str, required=True, help='[ ./data ] prefix path to data directory')
parser.add_argument('--output', type=str, required=True, help='[ ./output ] prefix path to output directory')
parser.add_argument('--dataset', type=str, default='HASC', help='dataset name ')
parser.add_argument('--loss', type=str, default='nce', help='loss function ')
parser.add_argument('--sim', type=str, default='cosine', help='similarity metric ')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

# hyperparameters for grid search
#parser.add_argument('--window', type=int, nargs='+', default=100, help='window size')
parser.add_argument('--window', type=int, default=100, help='window size')
parser.add_argument('--code', type=int, default=10, help='size of encoded features')
parser.add_argument('--beta', type=float, default=1, help='parameter for FN loss function or threshold for FC loss function')
parser.add_argument('--epoch', type=int, default=100, help='max iteration for training')
parser.add_argument('--batch', type=int, default=64, help='batch_size for training')
parser.add_argument('--eval_freq', type=int, default=25, help='evaluation frequency per batch updates')
parser.add_argument('--temp', type=float, default=.1, help='Temperature parameter for NCE loss function')
parser.add_argument('--tau', type=float, default=.1, help='parameter for Debiased contrastive loss function')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

# sanity check
args = parser.parse_args()
if not os.path.exists(os.path.join(args.output,args.dataset)):
    if not os.path.exists(args.output):
        os.mkdir(args.output)
        os.mkdir(os.path.join(args.output, "plots"))
        os.mkdir(os.path.join(args.output, "pred_sim"))
        os.mkdir(os.path.join(args.output, "model"))
    os.mkdir(os.path.join(args.output,args.dataset))
    os.mkdir(os.path.join(args.output,args.dataset, "plots"))
    os.mkdir(os.path.join(args.output,args.dataset, "pred_sim"))
    os.mkdir(os.path.join(args.output,args.dataset, "model"))



DATA_PATH = args.datapath
OUTPUT_PATH = os.path.join(args.output,args.dataset)
MODEL_PATH = os.path.join(args.output, "model")
DS_NAME = args.dataset
LOSS = args.loss
SIM = args.sim
GPU = args.gpu

WIN = args.window
CODE_SIZE = args.code
BATCH_SIZE = args.batch
EPOCHS = args.epoch
LR = args.lr
TEMP = args.temp
TAU = args.tau
BETA = args.beta
EVALFREQ = args.eval_freq

EPOCHS = EPOCHS * int(BATCH_SIZE / 4)
criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
decay_steps = 1000
lr_decayed_fn = tf.keras.experimental.CosineDecay(
        initial_learning_rate=LR, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.Adam(lr=LR)

train_name = "CP2_model_" + DS_NAME + "_T" + str(TEMP) + "_WIN" + str(WIN) + \
             "_BS" + str(BATCH_SIZE) + "_CS" + str(CODE_SIZE) + "_lr" + str(LR) + \
             "_LOSS" + LOSS +  "_SIM" + SIM + "_TAU" + str(TAU) + "_BETA" + str(BETA)
print("------------------------------------>>> " + train_name)

# -------------------------------
# 1 PREPARE DATASET
# -------------------------------
train_ds = load_dataset(DATA_PATH, DS_NAME, WIN, BATCH_SIZE, mode = "train")

# ------------------------
# 2 TRAINING
# ------------------------

prep_model = cp2.get_TCN_encoder((WIN,1), int(WIN / 2), CODE_SIZE)

if SIM == "cosine":
    similarity = ls._cosine_simililarity_dim2
# elif sim_fn == "dtw":
#    similarity = DTW
elif SIM == "euclidean":
    similarity = ls._euclidean_similarity_dim2
elif SIM == "edit":
    similarity = ls._edit_similarity_dim2
elif SIM == "nwarp":
    similarity = ls._neural_warp_dim2

epoch_wise_loss, epoch_wise_sim, epoch_wise_neg, prep_model = cp2.train_prep(prep_model, train_ds, OUTPUT_PATH, optimizer,
                                                                             criterion, train_name, WIN, temperature=TEMP,
                                                                             epochs=EPOCHS, sfn=similarity, lfn=LOSS, beta=BETA, tau= TAU)

# SAVE MODEL and Learning Progress plot
#with plt.xkcd():
splot=1
if splot ==1:
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(train_name)
    ax1.plot(epoch_wise_loss, label="Loss")
    ax2.plot(epoch_wise_sim, label="Positive pairs")
    ax2.plot(epoch_wise_neg, label="Negative pairs")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_PATH,"plots", LOSS+"__"+train_name + "_LOSS.png"))
    print("Learning progress plot saved!")


# -------------------------
# 4 TEST SET & SEGMENTATION
# -------------------------
test_ds = load_dataset(DATA_PATH, DS_NAME, WIN, BATCH_SIZE, mode = "test")
x_test, lbl_test = test_ds[:,1:], test_ds[:,0]

num = x_test.shape[0]
lbl_test = np.array(lbl_test).reshape((lbl_test.shape[0], 1))
history = prep_model(x_test[:, 0:WIN].reshape((num, 1, WIN)))
future = prep_model(x_test[:, WIN:].reshape((num, 1, WIN)))
pred_out = np.concatenate((lbl_test, history, future), 1)
rep_sim = ls._cosine_simililarity_dim1(history, future)

np.savetxt(os.path.join(OUTPUT_PATH, "pred_sim", train_name + "_pred_sim.csv"), np.concatenate((lbl_test, np.array(rep_sim).reshape((rep_sim.shape[0],1))),1), delimiter=',',
                   header="lbl,"+LOSS, comments="")
print("Saved test similarity result!")


print('Average similarity for test set : Reps : {}'.format(np.mean(rep_sim)))
gt = np.zeros(lbl_test.shape[0])
gt[np.where((lbl_test > int(2 * WIN * 0.15)) & (lbl_test < int(2 * WIN * 0.85)))[0]] = 1
# threshold_segmentation(h_pred,f_pred, gt, train_name, os.path.join(OUT_PATH,"Evaluation.txt"), threshold = np.mean(rep_sim) - np.std(rep_sim))
result = estimate_CPs(rep_sim, gt, os.path.join(OUTPUT_PATH, train_name),
                    os.path.join(OUTPUT_PATH, "Evaluation.txt"),
                    metric='cosine', threshold=epoch_wise_sim[-1] - ((epoch_wise_sim[-1]-epoch_wise_neg[-1])/3))
with open(os.path.join(OUTPUT_PATH, "Evaluation2.txt"), "a") as out_file:
    out_file.write(str(BATCH_SIZE) + "," + str(WIN) + "," + str(CODE_SIZE) + "," + str(TEMP) + "," + str(
                LR) + "," + str(np.mean(epoch_wise_loss))+ ","+str(epoch_wise_sim[-1]) + "," +str(epoch_wise_neg[-1])+","+result)
    out_file.close()
    print("Saved model to disk")
# -------------------------
# 3 SAVE THE MODEL
# -------------------------
prep_model.save_weights(os.path.join(MODEL_PATH, train_name + ".tf"))
model_json = prep_model.to_json()
with open(os.path.join(MODEL_PATH, train_name + ".json"), "w") as json_file:
    json_file.write(model_json)
    json_file.close()
    print("Saved model to disk")