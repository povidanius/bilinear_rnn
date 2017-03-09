import tensorflow as tf
import numpy as np
from linear import *




class EGRUCell(tf.contrib.rnn.RNNCell):
    """Extended Gated Recurrent Unit cell."""

    def __init__(self, num_units):
        #self._input_size = input_size
        self._num_units = num_units
      

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):

	batch_size = inputs.get_shape().as_list()[0]
	input_size = inputs.get_shape().as_list()[1]

        with tf.variable_scope(scope or type(self).__name__): 

          with tf.variable_scope("ir"):	# input and reset gates
                    ir = linear([inputs, state],  input_size + self._num_units, True, 1.0) 
		    ir = tf.nn.sigmoid(ir)
                    i, r = tf.split(value=ir, num_or_size_splits=[input_size, self._num_units], axis=1)
          with tf.variable_scope("uc"):  # update gate and candidate
                    uc =  linear([inputs, r * state], 2 * self._num_units, True)
                    u, c = tf.split(axis=1, num_or_size_splits=2, value=uc)
                    u = tf.nn.sigmoid(u)
		    c = tf.nn.tanh(c)

          new_h = u * state  + (1 - u) * c 

        return new_h, new_h

