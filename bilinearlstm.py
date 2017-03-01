import tensorflow as tf
import math
import numpy as np
from tensorflow.python.ops.rnn_cell import RNNCell
from helper import  *


class BilinearLSTM(tf.nn.rnn_cell.RNNCell):
'''
LSTM recurrent neural network with bilinear products.
Author: Povilas Daniusis, povilas.daniusis@gmail.com
https://github.com/povidanius/bilinear_rnn
'''

    def __init__(self, input_shape, hidden_shape):
        self._num_input_rows= input_shape[0]
        self._num_input_cols = input_shape[1]
        self._num_hidden_rows = hidden_shape[0]
        self._num_hidden_cols = hidden_shape[1]
        self._num_units = hidden_shape[0] * hidden_shape[1]
        self._num_perceptrons = tf.constant(4, tf.int32)

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units


    def __call__(self, inputs, state, scope=None):
        batch_size = inputs.get_shape()[0]
        c, h = state
        C = tf.reshape(c, [-1, self._num_hidden_rows, self._num_hidden_cols])
        H = tf.reshape(h, [-1, self._num_hidden_rows, self._num_hidden_cols])
        X = tf.reshape(inputs, [-1, self._num_rows, self._num_input_cols])

        with tf.variable_scope(scope or type(self).__name__):            


                    W1i = tf.get_variable("W1i",
                        shape=[self._num_hidden_rows, self._num_rows],
                        initializer=tf.truncated_normal_initializer())

                    W1f = tf.get_variable("W1f",
                        shape=[self._num_hidden_rows, self._num_rows],
                        initializer=tf.truncated_normal_initializer())

                    W1o = tf.get_variable("W1o",
                        shape=[self._num_hidden_rows, self._num_rows],
                        initializer=tf.truncated_normal_initializer())

                    W1c = tf.get_variable("W1c",
                        shape=[self._num_hidden_rows, self._num_rows],
                        initializer=tf.truncated_normal_initializer())


                    W2i = tf.get_variable("W2i",
                        shape=[self._num_input_cols, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())

                    W2f = tf.get_variable("W2f",
                        shape=[self._num_input_cols, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())

                    W2o = tf.get_variable("W2o",
                        shape=[self._num_input_cols, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())


                    W2c = tf.get_variable("W2c",
                        shape=[self._num_input_cols, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())



                    U1i = tf.get_variable("U1i",
                        shape=[self._num_hidden_rows, self._num_hidden_rows],
                        initializer=tf.truncated_normal_initializer())

                    U1f = tf.get_variable("U1f",
                        shape=[self._num_hidden_rows, self._num_hidden_rows],
                        initializer=tf.truncated_normal_initializer())


                    U1o = tf.get_variable("U1o",
                        shape=[self._num_hidden_rows, self._num_hidden_rows],
                        initializer=tf.truncated_normal_initializer())

                    U1c = tf.get_variable("U1c",
                        shape=[self._num_hidden_rows, self._num_hidden_rows],
                        initializer=tf.truncated_normal_initializer())




                    U2i = tf.get_variable("U2i",
                        shape=[self._num_hidden_cols, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())

                    U2f = tf.get_variable("U2f",
                        shape=[self._num_hidden_cols, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())

                    U2o = tf.get_variable("U2o",
                        shape=[self._num_hidden_cols, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())

                    U2c = tf.get_variable("U2c",
                        shape=[self._num_hidden_cols, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())


                  
                    Bi = tf.get_variable("Bi",
                        shape=[self._num_hidden_rows, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())

                    Bf = tf.get_variable("Bf",
                        shape=[self._num_hidden_rows, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())

                    Bo = tf.get_variable("Bo",
                        shape=[self._num_hidden_rows, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())


                    Bc = tf.get_variable("Bc",
                        shape=[self._num_hidden_rows, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())


                    I = tf.nn.sigmoid(dot(tf.transpose(dot(W1i, X), [1, 0, 2]), W2i) + dot(tf.transpose(dot(U1i, H), [1, 0, 2]), U2i) + Bi)
                    F = tf.nn.sigmoid(dot(tf.transpose(dot(W1f, X), [1, 0, 2]), W2f) + dot(tf.transpose(dot(U1f, H), [1, 0, 2]), U2f) + Bf)
                    F = tf.nn.sigmoid(dot(tf.transpose(dot(W1o, X), [1, 0, 2]), W2o) + dot(tf.transpose(dot(U1o, H), [1, 0, 2]), U2o) + Bo)
                    C_tilde = tf.nn.tanh(dot(tf.transpose(dot(W1c, X), [1, 0, 2]), W2c) + dot(tf.transpose(dot(U1c, H), [1, 0, 2]), U2c) + Bc)

                    
                    C_new = I * C_tilde + F * C
                    H_new = O * tf.nn.tanh(C_new)

                    new_state = tf.nn.rnn_cell.LSTMStateTuple(tf.reshape(C_new, [-1, self._num_hidden_rows* self._num_hidden_cols]), tf.reshape(H_new, [-1, self._num_hidden_rows* self._num_hidden_cols]))

        return new_h, new_state


# --------- testing of update step -------------
'''
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

NB = 100
DX1 = 5
DX2 = 4
DH1 = 6
DH2 = 7
NGATES = 4
X = tf.Variable(tf.random_normal([NB,DX1,DX2]))
W1 = tf.Variable(tf.random_normal([NGATES * DH1, DX1]))
W2 = tf.Variable(tf.random_normal([NGATES * DX2, DH2]))
B = tf.Variable(tf.random_normal([NGATES, DH1, DH2]))

print ("W1 variable shape {}".format(W1.get_shape()))


print ("W2 variable shape {}".format(W2.get_shape()))
W2 = tf.tile(W2, tf.pack([NB, 1]))
W2 = tf.reshape(W2, [NB * NGATES, DX2, DH2])
print ("W2 shape {}".format(W2.get_shape()))


P1 = tf.transpose(dot(W1, X), [1, 0, 2])
print("P1 variable shape {}".format(P1.get_shape()))
P1 = tf.reshape(P1, [NB * NGATES, DH1, DX2])
print("P1 shape {}".format(P1.get_shape()))

print("B shape {}".format(B.get_shape()))
B = tf.tile(B, tf.pack([NB, 1, 1]))
print("B shape {}".format(B.get_shape()))

IFOC = tf.nn.sigmoid(tf.batch_matmul(P1, W2) +  B)
print("IFOC shape {}".format(IFOC.get_shape()))
'''








