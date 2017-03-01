import tensorflow as tf
import math
import numpy as np
from tensorflow.python.ops.rnn_cell import RNNCell
from helper import  *

class BilinearGRU(tf.nn.rnn_cell.RNNCell):
'''
GRU recurrent neural network with bilinear dot products.
Author: Povilas Daniusis, povilas.daniusis@gmail.com
https://github.com/povidanius/bilinear_rnn
'''

    def __init__(self, input_shape, hidden_shape):
        self._num_input_rows= input_shape[0]
        self._num_input_cols = input_shape[1]
        self._num_hidden_rows = hidden_shape[0]
        self._num_hidden_cols = hidden_shape[1]
        self._num_units = hidden_shape[0] * hidden_shape[1]
        #self._num_perceptrons = tf.constant(4, tf.int32)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units


    def __call__(self, inputs, state, scope=None):
        batch_size = inputs.get_shape()[0]    
        H = tf.reshape(state, [-1, self._num_hidden_rows, self._num_hidden_cols])
        X = tf.reshape(inputs, [-1, self._num_rows, self._num_input_cols])

        with tf.variable_scope(scope or type(self).__name__):            


                    W1u = tf.get_variable("W1u",
                        shape=[self._num_hidden_rows, self._num_rows],
                        initializer=tf.truncated_normal_initializer())

                    W2u = tf.get_variable("W2u",
                        shape=[self._num_input_cols, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())


                    W1r = tf.get_variable("W1r",
                        shape=[self._num_hidden_rows, self._num_rows],
                        initializer=tf.truncated_normal_initializer())

                    W2r = tf.get_variable("W2r",
                        shape=[self._num_input_cols, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())


                    W1h = tf.get_variable("W1h",
                        shape=[self._num_hidden_rows, self._num_rows],
                        initializer=tf.truncated_normal_initializer())

                    W2h = tf.get_variable("W2h",
                        shape=[self._num_input_cols, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())


                    U1u = tf.get_variable("U1u",
                        shape=[4*self._num_hidden_rows, self._num_hidden_rows],
                        initializer=tf.truncated_normal_initializer())

                    U2u = tf.get_variable("U2u",
                        shape=[self._num_hidden_cols, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())

                    U1r = tf.get_variable("U1r",
                        shape=[4*self._num_hidden_rows, self._num_hidden_rows],
                        initializer=tf.truncated_normal_initializer())

                    U2r = tf.get_variable("U2r",
                        shape=[self._num_hidden_cols, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())

                    U1h = tf.get_variable("U1h",
                        shape=[4*self._num_hidden_rows, self._num_hidden_rows],
                        initializer=tf.truncated_normal_initializer())

                    U2h = tf.get_variable("U2h",
                        shape=[self._num_hidden_cols, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())

                    Bu = tf.get_variable("Bu",
                        shape=[self._num_hidden_rows, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())

                    Br = tf.get_variable("Br",
                        shape=[self._num_hidden_rows, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())

                    Bh = tf.get_variable("Bh",
                        shape=[self._num_hidden_rows, self._num_hidden_cols],
                        initializer=tf.truncated_normal_initializer())


                    # -----------------------
                    U = tf.nn.sigmoid(dot(tf.transpose(dot(W1u, X), [1, 0, 2]), W2u) + dot(tf.transpose(dot(U1u, H), [1, 0, 2]), U2u) + Bu)
                    R = tf.nn.sigmoid(dot(tf.transpose(dot(W1r, X), [1, 0, 2]), W2r) + dot(tf.transpose(dot(U1r, H), [1, 0, 2]), U2r) + Br)
                    H_tilde = dot(tf.transpose(dot(W1h, X), [1, 0, 2]), W2h) + R * dot(tf.transpose(dot(U1h, H), [1, 0, 2]), U2h) + Bh
  
                    H_new = U * tf.nn.tanh(Htilde) + (tf.ones_like(U) - U) * H 
                    new_state = tf.reshape(H_new, [-1, self._num_hidden_rows* self._num_hidden_cols])

        return new_state

'''
# --------- testing of update step -------------

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

NB = 100
DX1 = 5
DX2 = 4
DH1 = 6
DH2 = 7
NGATES = 4
X = tf.Variable(tf.random_normal([NB,DX1,DX2]))
W1 = tf.Variable(tf.random_normal([DH1, DX1]))
W2 = tf.Variable(tf.random_normal([DX2, DH2]))

W1XW2 = dot(tf.transpose(dot(W1, X), [1, 0, 2]), W2)

print("W1XW2 shape {}".format(W1XW2.get_shape()))
'''




