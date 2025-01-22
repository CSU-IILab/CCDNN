from keras.layers import Input, Dense, Dropout, Flatten, GRU, BatchNormalization,Softmax,Conv1D,Conv2D,MaxPooling2D,UpSampling2D,MaxPooling1D,UpSampling1D
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
import math
import numpy as np
import tensorflow as tf

def my_act(x):
    return (x ** 3) / 3 + x

def my_init_sigmoid(shape, dtype=None):
    rnd = K.random_uniform(
        shape, 0., 1., dtype)
    from keras.initializers import _compute_fans
    fan_in, fan_out = _compute_fans(shape)
    return 8. * (rnd - 0.5) * math.sqrt(6) / math.sqrt(fan_in + fan_out)
    
def Cal_Loss_MSE(y_true,y_pred):       
    squared_difference = tf.square(y_true - y_pred)
    mse = tf.reduce_mean(squared_difference)
    return mse
    
def my_init_others(shape, dtype=None):
    rnd = K.random_uniform(
        shape, 0., 1., dtype)
    from keras.initializers import _compute_fans
    fan_in, fan_out = _compute_fans(shape)
    return 2. * (rnd - 0.5) / math.sqrt(fan_in)

class DeepCCA_cnn():
    def __init__(self, input_size1,
                 input_size2, outdim_size, class_num, reg_par, time_step, RF_flag,coder_style):
        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.outdim_size = outdim_size
        self.class_num = class_num
        self.reg_par = reg_par
        self.time_step = time_step
        self.RF_flag = RF_flag
        self.coder_style = coder_style
        self.input_view1 = tf.placeholder(tf.float32, [None, time_step, input_size1])
        self.input_view2 = tf.placeholder(tf.float32, [None, time_step, input_size2])
        self.label_placeholder = tf.placeholder(tf.float32, shape=[None, class_num])
        self.rate = tf.placeholder(tf.float32)
        self.num_sample = 256
        
        self.encoder_view1, self.output_view1 = self.layers1()
        self.encoder_view2, self.output_view2 = self.layers2()
                 
        print(self.input_view1.shape)
        print(self.encoder_view1.shape)
        print(self.output_view1.shape)
        
        self.neg_corr, self.S = self.neg_correlation(self.output_view1, self.output_view2, self.class_num)
        
        if self.RF_flag:
            self.res1,self.res2 = self.Cal_residual(self.S, self.output_view1, self.output_view2, self.outdim_size, self.num_sample)
            self.decoder_view1 = self.layers6()
            print(self.decoder_view1.shape)
            self.decoder_view2 = self.layers7()
        else:
            self.res1,self.res2 = self.output_view1, self.output_view2
            self.decoder_view1 = self.layers6()
            print(self.decoder_view1.shape)
            self.decoder_view2 = self.layers7()
        self.Pearson = self.cal_pearson()
        self.loss_MSE = Cal_Loss_MSE(self.input_view1, self.decoder_view1)+Cal_Loss_MSE(self.input_view2, self.decoder_view2)
        self.loss_MAE = tf.reduce_mean(tf.abs(self.input_view1 - self.decoder_view1)) + tf.reduce_mean(tf.abs(self.input_view2 - self.decoder_view2)) 

    def layers1(self):
        if self.coder_style == 'CNN':
            input1 = self.input_view1
            layer1_1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input1)
            layer1_2 = MaxPooling1D(2)(layer1_1)
            layer1_3 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(layer1_2)
            encoded_1 = MaxPooling1D(2)(layer1_3)

            layer1_5 = Flatten()(encoded_1)
            layer1_6 = Dense(64, activation=tf.nn.relu, kernel_initializer=my_init_sigmoid,
                             kernel_regularizer=l2(self.reg_par))(layer1_5)
            out1 = Dense(self.outdim_size, activation=None, kernel_initializer=my_init_others,
                         kernel_regularizer=l2(self.reg_par))(layer1_6)
        elif self.coder_style == 'FC':
            input1 = self.input_view1
            layer1_1 = Dense(256, activation=tf.nn.relu, kernel_initializer=my_init_sigmoid,
                             kernel_regularizer=l2(self.reg_par))(input1)
            layer1_2 = MaxPooling1D(2)(layer1_1)
            layer1_3 = Dense(16, activation=tf.nn.relu,kernel_initializer=my_init_sigmoid, kernel_regularizer=l2(self.reg_par))(layer1_2)
            encoded_1 = MaxPooling1D(2)(layer1_3)
            layer1_5 = Flatten()(encoded_1)
            layer1_6 = Dense(64, activation=tf.nn.relu, kernel_initializer=my_init_sigmoid,
                             kernel_regularizer=l2(self.reg_par))(layer1_5)
            out1 = Dense(self.outdim_size, activation=None, kernel_initializer=my_init_others,
                         kernel_regularizer=l2(self.reg_par))(layer1_6)
            
        return encoded_1,out1

    def layers2(self):
        if self.coder_style == 'CNN':
            input2 = self.input_view2
            layer2_1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input2)
            layer2_2 = MaxPooling1D(2, padding='same')(layer2_1)
            layer2_3 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(layer2_2)
            encoded_2 = MaxPooling1D(2, padding='same')(layer2_3)

            layer2_5 = Flatten()(encoded_2)
            layer2_6 = Dense(64, activation=tf.nn.relu, kernel_initializer=my_init_sigmoid,
                             kernel_regularizer=l2(self.reg_par))(layer2_5)
            out2 = Dense(self.outdim_size, activation=None, kernel_initializer=my_init_others,
                         kernel_regularizer=l2(self.reg_par))(layer2_6)

        elif self.coder_style == 'FC':
            input2 = self.input_view2
            layer2_1 = Dense(256, activation=tf.nn.relu, kernel_initializer=my_init_sigmoid,
                             kernel_regularizer=l2(self.reg_par))(input2)
            layer2_2 = MaxPooling1D(2)(layer2_1)
            layer2_3 = Dense(16, activation=tf.nn.relu,kernel_initializer=my_init_sigmoid, kernel_regularizer=l2(self.reg_par))(layer2_2)
            encoded_2 = MaxPooling1D(2)(layer2_3)
            layer2_5 = Flatten()(encoded_2)
            layer2_6 = Dense(64, activation=tf.nn.relu, kernel_initializer=my_init_sigmoid,
                             kernel_regularizer=l2(self.reg_par))(layer2_5)
            out2 = Dense(self.outdim_size, activation=None, kernel_initializer=my_init_others,
                         kernel_regularizer=l2(self.reg_par))(layer2_6)
        return encoded_2,out2
    
    def layers6(self):
        if self.coder_style == 'FC':
            input6 = self.res1
            layer6_1 = Dense(256, activation=tf.nn.relu, kernel_initializer=my_init_sigmoid,
                             kernel_regularizer=l2(self.reg_par))(input6)
            layer6_2 = Dense(784, activation=tf.nn.relu, kernel_initializer=my_init_sigmoid,
                             kernel_regularizer=l2(self.reg_par))(layer6_1)
            decoded_1 = tf.reshape(layer6_2,[-1,784,1])
        elif self.coder_style == 'CNN':
            input6 = self.res1
            layer6_1 = Dense(self.outdim_size*2, activation=tf.nn.relu, kernel_initializer=my_init_sigmoid,
                             kernel_regularizer=l2(self.reg_par))(input6)
            layer6_2 = Dense(196*16, activation=tf.nn.relu, kernel_initializer=my_init_sigmoid,
                             kernel_regularizer=l2(self.reg_par))(layer6_1)
            layer6_2 = tf.reshape(layer6_2,[-1,196,16])
            layer6_3 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(layer6_2)
            layer6_4 = UpSampling1D(2)(layer6_3)
            layer6_5 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(layer6_4)
            layer6_6 = UpSampling1D(2)(layer6_5)
            decoded_1 = Conv1D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(layer6_6)
        return decoded_1
    
    def layers7(self):
        if self.coder_style == 'FC':
            input7 = self.res2
            layer7_1 = Dense(256, activation=tf.nn.relu, kernel_initializer=my_init_sigmoid,
                             kernel_regularizer=l2(self.reg_par))(input7)
            layer7_2 = Dense(784, activation=tf.nn.relu, kernel_initializer=my_init_sigmoid,
                             kernel_regularizer=l2(self.reg_par))(layer7_1)
            decoded_2 = tf.reshape(layer7_2,[-1,784,1])
        elif self.coder_style == 'CNN':
            input7 = self.res2
            layer7_1 = Dense(self.outdim_size*2, activation=tf.nn.relu, kernel_initializer=my_init_sigmoid,
                             kernel_regularizer=l2(self.reg_par))(input7) 
            layer7_2 = Dense(196*16, activation=tf.nn.relu, kernel_initializer=my_init_sigmoid,
                             kernel_regularizer=l2(self.reg_par))(layer7_1) 
            layer7_2 = tf.reshape(layer7_2,[-1,196,16])
            layer7_3 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(layer7_2)
            layer7_4 = UpSampling1D(2)(layer7_3)
            layer7_5 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(layer7_4)
            layer7_6 = UpSampling1D(2)(layer7_5)
            decoded_2 = Conv1D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(layer7_6)
        return decoded_2
    
    def cal_pearson(self):

        array1 = self.output_view1
        array2 = self.output_view2

        mean1 = tf.reduce_mean(array1, axis=1, keepdims=True)
        mean2 = tf.reduce_mean(array2, axis=1, keepdims=True)

        covariance_matrix = tf.reduce_mean((array1 - mean1) * (array2 - mean2), axis=1, keepdims=True)

        stddev1 = tf.math.reduce_std(array1, axis=1, keepdims=True)
        stddev2 = tf.math.reduce_std(array2, axis=1, keepdims=True)

        pearson_correlation = covariance_matrix / (stddev1 * stddev2)
        pearson_correlation = tf.reduce_mean(pearson_correlation)
        return pearson_correlation
    
    def neg_correlation(self, output1, output2, class_num):
        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-12

        H1 = tf.transpose(output1)
        H2 = tf.transpose(output2)

        m = tf.shape(H1)[1]

        H1bar = H1 - (1.0 / tf.cast(m, tf.float32)) * tf.matmul(H1, tf.ones([m, m]))
        H2bar = H2 - (1.0 / tf.cast(m, tf.float32)) * tf.matmul(H2, tf.ones([m, m]))

        SigmaHat12 = (1.0 / (tf.cast(m, tf.float32) - 1)) * tf.matmul(H1bar, tf.transpose(H2bar))
        SigmaHat11 = (1.0 / (tf.cast(m, tf.float32) - 1)) * tf.matmul(H1bar, tf.transpose(H1bar)) + r1 * tf.eye(
            self.outdim_size)
        SigmaHat22 = (1.0 / (tf.cast(m, tf.float32) - 1)) * tf.matmul(H2bar, tf.transpose(H2bar)) + r2 * tf.eye(
            self.outdim_size)

        [D1, V1] = tf.self_adjoint_eig(SigmaHat11)
        [D2, V2] = tf.self_adjoint_eig(SigmaHat22)

        posInd1 = tf.where(tf.greater(D1, eps))
        posInd1 = tf.reshape(posInd1, [-1, tf.shape(posInd1)[0]])[0]
        D1 = tf.gather(D1, posInd1)
        V1 = tf.gather(V1, posInd1)

        posInd2 = tf.where(tf.greater(D2, eps))
        posInd2 = tf.reshape(posInd2, [-1, tf.shape(posInd2)[0]])[0]
        D2 = tf.gather(D2, posInd2)
        V2 = tf.gather(V2, posInd2)

        SigmaHat11RootInv = tf.matmul(tf.matmul(V1, tf.linalg.diag(D1 ** -0.5)), tf.transpose(V1))
        SigmaHat22RootInv = tf.matmul(tf.matmul(V2, tf.linalg.diag(D2 ** -0.5)), tf.transpose(V2))

        Tval = tf.matmul(tf.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)
        Tval.set_shape([self.outdim_size, self.outdim_size])
        s = tf.svd(Tval, compute_uv=False)

        [U, V] = tf.self_adjoint_eig(tf.matmul(Tval, Tval, transpose_a=True))
        idx3 = tf.where(U > eps)[:, 0]
        dim_svd = tf.cond(tf.size(idx3) < self.outdim_size, lambda: tf.size(idx3), lambda: self.outdim_size)
        corr = tf.reduce_sum(tf.sqrt(U[-dim_svd:]))

        return -corr, s
    
    def Cal_residual(self, S_up, hidden_view1_up, hidden_view2_up, outdim_size, num_sample):
        eps = 1e-8
        S2_up = tf.linalg.diag(S_up)

        hidden1 = tf.transpose(hidden_view1_up)
        hidden2 = tf.transpose(hidden_view2_up)

        J = tf.eye(outdim_size)
        L = tf.eye(outdim_size)
        JT = tf.transpose(J)
        LT = tf.transpose(L)

        r1 = tf.matmul(JT, hidden1) - tf.matmul(S2_up, tf.matmul(LT, hidden2))
        r2 = tf.matmul(LT, hidden2) - tf.matmul(S2_up, tf.matmul(JT, hidden1))

        JTInv = tf.linalg.pinv(JT + eps * tf.eye(tf.shape(JT)[0]))
        LTInv = tf.linalg.pinv(LT + eps * tf.eye(tf.shape(LT)[0]))

        r1 = tf.matmul(JTInv, r1)
        r2 = tf.matmul(LTInv, r2)

        res1_up = tf.transpose(r1)
        res2_up = tf.transpose(r2)

        return res1_up, res2_up
