import tensorflow as tf
import math
import numpy as np
from helper import  *


class BilinearGRU(tf.contrib.rnn.RNNCell):

    def __init__(self, input_shape, hidden_shape):
        self._num_input_rows= input_shape[0]
        self._num_input_cols = input_shape[1]
        self._num_hidden_rows = hidden_shape[0]
        self._num_hidden_cols = hidden_shape[1]
        self._num_units = hidden_shape[0] * hidden_shape[1]


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units


    def __call__(self, inputs, state, scope=None):

        batch_size = inputs.get_shape().as_list()[0]

	#print("batch_size {}".format(batch_size))

        H = tf.reshape(state, [-1, self._num_hidden_rows, self._num_hidden_cols])
        X = tf.reshape(inputs, [-1, self._num_input_rows, self._num_input_cols])

        with tf.variable_scope(scope or type(self).__name__):            


                    W1u = tf.get_variable("W1u",
                        shape=[self._num_hidden_rows, self._num_input_rows])

                    W2u = tf.get_variable("W2u",
                        shape=[self._num_input_cols, self._num_hidden_cols])


                    W1r = tf.get_variable("W1r",
                        shape=[self._num_hidden_rows, self._num_input_rows])

                    W2r = tf.get_variable("W2r",
                        shape=[self._num_input_cols, self._num_hidden_cols])


                    W1h = tf.get_variable("W1h",
                        shape=[self._num_hidden_rows, self._num_input_rows])

                    W2h = tf.get_variable("W2h",
                        shape=[self._num_input_cols, self._num_hidden_cols])


                    U1u = tf.get_variable("U1u",
                        shape=[self._num_hidden_rows, self._num_hidden_rows])

                    U2u = tf.get_variable("U2u",
                        shape=[self._num_hidden_cols, self._num_hidden_cols])

                    U1r = tf.get_variable("U1r",
                        shape=[self._num_hidden_rows, self._num_hidden_rows])

                    U2r = tf.get_variable("U2r",
                        shape=[self._num_hidden_cols, self._num_hidden_cols])

                    U1h = tf.get_variable("U1h",
                        shape=[self._num_hidden_rows, self._num_hidden_rows])

                    U2h = tf.get_variable("U2h",
                        shape=[self._num_hidden_cols, self._num_hidden_cols])

                    Bu = tf.get_variable("Bu",
                        shape=[self._num_hidden_rows, self._num_hidden_cols])

                    Br = tf.get_variable("Br",
                        shape=[self._num_hidden_rows, self._num_hidden_cols])

                    Bh = tf.get_variable("Bh",
                        shape=[self._num_hidden_rows, self._num_hidden_cols])


                    U = tf.nn.sigmoid( dot(tf.transpose(dot(W1u, X), [1, 0, 2]), W2u) + dot(tf.transpose(dot(U1u, H), [1, 0, 2]), U2u) + Bu)
                    R = tf.nn.sigmoid(dot(tf.transpose(dot(W1r, X), [1, 0, 2]), W2r) + dot(tf.transpose(dot(U1r, H), [1, 0, 2]), U2r) + Br)
                    H_tilde = dot(tf.transpose(dot(W1h, X), [1, 0, 2]), W2h) + R * dot(tf.transpose(dot(U1h, H), [1, 0, 2]), U2h) + Bh
  
                    H_new = U * tf.nn.tanh(H_tilde) + (tf.ones_like(U) - U) * H
                    new_state = tf.reshape(H_new, [-1, self._num_hidden_rows*self._num_hidden_cols])

        return new_state, new_state




