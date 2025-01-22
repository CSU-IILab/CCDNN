import os
import time
import random
from keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import shutil
from scipy.io import savemat
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()
import numpy as np
import argparse
from matplotlib import rcParams
config = {
    "font.family": 'stix',
    "font.size": 15,
}
rcParams.update(config)
from datetime import datetime

parser = argparse.ArgumentParser(description='train model')
parser.add_argument('--model', type=str, default = 'GRU')
parser.add_argument('--model-cca', type=str, default = 'CCDNN')
parser.add_argument('--train', type=str, default = 'False')
parser.add_argument('--epoch', type=int, default = 2)
parser.add_argument('--window-len', type=int, default = 28)
parser.add_argument('--output-dim', type=int, default = 50)
parser.add_argument('--model-index', type=int, default = 1)
parser.add_argument('--RF-flag', type=str, default = 'True')
parser.add_argument('--coder-style', type=str, default = 'CNN')

args = parser.parse_args()
model = args.model
model_select = args.model_cca
retrain = True if args.train == 'True' else False
n_epochs = args.epoch
batch_len = args.window_len
outdim_size = args.output_dim
model_index = args.model_index
RF_flag = args.RF_flag
coder_style = args.coder_style
if model == 'cnn':
    from model_cnn import DeepCCA_cnn as DeepCCA_model
# elif model == 'lstm':
#     from model_lstm import DeepCCA_lstm as DeepCCA_model
# elif model == 'gru':
#     from model_gru import DeepCCA_gru as DeepCCA_model
# elif model == 'attention':
#     from model_attention import DeepCCA_attention as DeepCCA_model


saveName = model
model_save_rootpath = './model'
dir_path = os.path.join(model_save_rootpath, f'{model}_{batch_len}_{outdim_size}_{model_index}')
if (retrain == True):
    try:
        shutil.rmtree(dir_path)
    except:
        pass
model_save_path = os.path.join(dir_path, "best.ckpt")
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

log_save_rootpath = './model/'
dir_path = os.path.join(log_save_rootpath, f'log_{model}_{batch_len}_{outdim_size}_{model_index}')
if (retrain == True):
    try:
        shutil.rmtree(dir_path)
    except:
        pass
log_save_path = dir_path
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

num_classes = 10 

# Load MNIST dataset
input_batch_train = np.load('./dataset/Minist/train_view1.npy').reshape(-1, 28*28, 1)
output_batch_train = np.load('./dataset/Minist/train_view2.npy').reshape(-1, 28*28, 1)
label_train = np.load('./dataset/Minist/train_label.npy')
label_train = to_categorical(label_train, num_classes)

input_batch_test = np.load('./dataset/Minist/test_view1.npy').reshape(-1, 28*28, 1)
output_batch_test = np.load('./dataset/Minist/test_view2.npy').reshape(-1, 28*28, 1)
label_test = np.load('./dataset/Minist/test_label.npy')
label_test = to_categorical(label_test, num_classes)

# Convert one-dimensional labels to one-hot encoding
class_num = label_train.shape[1]

learning_rate = 0.001
momentum = 0.5
batch_size = 256
num_sample = 6000  # Number of data subset
input_size1 = input_batch_train.shape[2]  # Number of input channels
input_size2 = output_batch_train.shape[2]  # Number of output channels
input_2d = input_batch_train.shape[1]
reg_par = 2
use_all_singular_values = True

tf.reset_default_graph()
rand_num = random.randint(0, 1000)
K.set_learning_phase(1)
dcca_model = DeepCCA_model(input_size1, input_size2, outdim_size, class_num, reg_par, input_2d, RF_flag, coder_style)

input_view1 = dcca_model.input_view1
input_view2 = dcca_model.input_view2
label_placeholder = dcca_model.label_placeholder
hidden_view1 = dcca_model.output_view1
hidden_view2 = dcca_model.output_view2
lamda = 0.001
loss_MSE =  dcca_model.loss_MSE
loss_MAE =  dcca_model.loss_MAE
neg_corr = dcca_model.neg_corr
res1 = dcca_model.res1
S = dcca_model.S
rate_ = dcca_model.rate
num_sample_1 = dcca_model.num_sample
Pearson = dcca_model.Pearson
decoder_view1 = dcca_model.decoder_view1
decoder_view2 = dcca_model.decoder_view2

if model_select == 'CCDNN':
    loss = dcca_model.loss_MSE
    
tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()

# Define optimizer
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

saver = tf.train.Saver()
loss_now = 0
last_loss = 0
loss_his = []

corr1 = []
corr2 = []
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3


def draw_pic(x, PIC):
    size1 = int(x.shape[1]** 0.5)
    x = x.reshape(-1, size1, size1)
    image_array = x

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(3):
        for j in range(3):
            index = i * 3 + j
            ax = axes[i, j]
            image_data = image_array[index]
            ax.imshow(image_data, cmap='gray')
            ax.axis('off')
    plt.tight_layout()
    plt.show()
    
os.environ["CUDA_VISIBLE_DEVICES"] = "1"     
with tf.Session(config=config) as sess:

    if not os.path.exists(model_save_path + ".index"):
        print("Learning begins...")

        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(log_save_path, sess.graph)


        start_time = time.time()
        epoch_list = []
        loss_list = []
        cor_test_list = []
        mae_test_list = []
        mse_test_list = []
        pearson_test_list = []
        for epoch in range(n_epochs):
            start_time = time.time()
            epoch_list.append(epoch+1)
            iterations = 0
            print(f'epoch {epoch + 1}')
            current_loss_total = 0
            Pearson_total_test = 0
            for batch in range(int(input_batch_train.shape[0]/batch_size)):
                iterations+=1
                X1_batch = input_batch_train[(batch * batch_size): ((batch + 1) * batch_size)]
                X2_batch = output_batch_train[(batch * batch_size): ((batch + 1) * batch_size)]
                label_batch = label_train[(batch * batch_size): ((batch + 1) * batch_size)]
                _, current_loss,current_loss_MSE, current_MAE, summary, current_Pearson  = sess.run([train_op, loss, loss_MSE, loss_MAE, merged,Pearson],feed_dict={input_view1: X1_batch, input_view2: X2_batch,label_placeholder: label_batch})
                train_writer.add_summary(summary, batch + epoch * batch_size)
                loss_now = current_loss_MSE
                loss_his.append(loss_now)
                if last_loss > loss_now:
                    last_loss = loss_now
                    if not os.path.exists(model_save_path):
                        os.mkdir(model_save_path)
                        print("create the directory: %s" % model_save_path)
                    saver.save(sess, model_save_path)
                current_loss_total += current_loss_MSE
                current_loss_all = 0
                current_neg_corr_all = 0
                current_mae_all = 0
                if iterations == int(input_batch_train.shape[0]/batch_size-1):
                    for batch in range(int(input_batch_test.shape[0]/batch_size)):
                        X1_batch_test = input_batch_test[int(batch * batch_size): int((batch + 1) * batch_size)]
                        X2_batch_test = output_batch_test[int(batch * batch_size): int((batch + 1) * batch_size)]
                        label_batch_test = label_test[int(batch * batch_size): int((batch + 1) * batch_size)]
                        current_loss_test,current_loss_MSE_test,current_loss_MAE_test,current_neg_corr, current_Pearson_test,current_decoder_view1_test,current_decoder_view2_test  = sess.run([loss,loss_MSE,loss_MAE, neg_corr,Pearson,decoder_view1,decoder_view2],feed_dict={input_view1: X1_batch_test, input_view2: X2_batch_test,label_placeholder: label_batch_test})
                        Pearson_total_test += current_Pearson_test
                        current_loss_all += current_loss_MSE_test
                        current_mae_all += current_loss_MAE_test
                        current_neg_corr_all += current_neg_corr
                    pic_name = ['normal_input.png','noisy_input.png','normal_output.png','noisy_output.png']
                    draw_pic(X1_batch_test[1:10],pic_name[0])
                    draw_pic(X2_batch_test[1:10],pic_name[1])
                    draw_pic(current_decoder_view1_test[1:10],pic_name[2])
                    draw_pic(current_decoder_view2_test[1:10],pic_name[3])
                    time_node = time.time()
                    print("epoch: " + str(epoch) + " time costs " + str(time_node - start_time) + "s")
                    print(f'epoch {epoch+1}, Loss_test (mae): {current_loss_all/(batch+1)}')
                    print(f'epoch {epoch+1}, Mse_test: {current_mae_all/(batch+1)}')
                    print("epoch: " + str(epoch+1) + " Pearson correlation for test:", Pearson_total_test/((batch+1)))
                    
                    cor_test_list.append(-current_neg_corr_all/((batch+1)))
                    mae_test_list.append(current_mae_all/(batch+1))
                    mse_test_list.append(current_loss_all/(batch+1))
                    pearson_test_list.append(Pearson_total_test/((batch+1)))
            loss_list.append(current_loss_total/(iterations + 1)) 
        fig, ax1 = plt.subplots()

        ax1.plot(epoch_list, loss_list, color='blue', linestyle='-', marker='o',label = 'Loss')
        ax1.plot(epoch_list, cor_test_list, color='darkorange', linestyle='-', marker='o',label = 'Total Correlation ')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss and Total Correlation Value', color='black')
        ax1.set_xlim((1, epoch_list[-1]))
        ax1.set_ylim((0,1))
        ax1.tick_params('y', colors='black')
        
        plt.title('Training process')
        fig.tight_layout()

        fig.legend(bbox_to_anchor=(1.00, 0.25), loc=3, borderaxespad=0)
        plt.show()
        print("The min mse_test is "+ str(min(mse_test_list))+"，Located in Epoch " +str(mse_test_list.index(min(mse_test_list))))
        print("The min mae_test is "+ str(min(mae_test_list))+"，Located in Epoch " +str(mae_test_list.index(min(mae_test_list))))
        print("The max cor_test is "+ str(max(cor_test_list))+"，Located in Epoch " +str(cor_test_list.index(max(cor_test_list))))
        print("The max cor_pearson is "+ str(max(pearson_test_list))+"，Located in Epoch " +str(pearson_test_list.index(max(pearson_test_list))))