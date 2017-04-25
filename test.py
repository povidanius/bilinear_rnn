'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

import matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt


import cv2, requests, numpy
import time
from egrucell import *
#from supercell import *
from bilinear_rnn import *
#from multicelllstm import *



import tensorflow as tf

#from tensorflow.python.ops import control_flow_ops
#from datasets import dataset_factory
#from deployment import model_deploy
#from nets import nets_factory
#from preprocessing import preprocessing_factory

#slim = tf.contrib.slim
#slim = tf.contrib.slim
#from slim.datasets import download_and_convert_cifar10

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


def my_cnn(images, num_classes, is_training):  # is_training is not used...
    with slim.arg_scope([slim.max_pool2d], kernel_size=[3, 3], stride=2):
        net = slim.conv2d(images, 64, [5, 5])
        net = slim.max_pool2d(net)
        net = slim.conv2d(net, 64, [5, 5])
        net = slim.max_pool2d(net)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 192)
        net = slim.fully_connected(net, num_classes, activation_fn=None)       
        return net




mnist = input_data.read_data_sets("data", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 0.5 * 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 2*28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    #rnn_cell = BilinearLSTM(input_shape = [7,4], hidden_shape = [32, 4])

    rnn_cell = EGRUCell(n_hidden)
    #rnn_cell = rnn.GRUCell(n_hidden)

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)

    
    #print (len(x))
   

    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
   
    #print (len(outputs))
    #print(outputs[-1].get_shape())
    #print(states.get_shape())


    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def prepare_batch(batch_x):
  batch_x_t = np.transpose(batch_x, [0, 2, 1])
  batch_x_new = np.concatenate([batch_x, batch_x_t], 1)
 
  return batch_x_new


pred = RNN(x, weights, biases)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
	

        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, 28, n_input))

	batch_x = prepare_batch(batch_x)

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")


    test_len = 40000
    test_data = mnist.test.images[:test_len].reshape((-1, 28, n_input))    
    test_data = prepare_batch(test_data)

    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))



# --------- testing of update step -------------

"""
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

NB = 100
DX1 = 7
DX2 = 4
DH1 = 32
DH2 = 7
NGATES = 4
X = tf.Variable(tf.random_normal([NB,DX1,DX2]))
W1 = tf.Variable(tf.random_normal([DH1, DX1]))
W2 = tf.Variable(tf.random_normal([DX2, DH2]))

W1X = dot(W1,X)
print("W1X shape {}".format(W1X.get_shape()))
"""


#W1XW2 = dot(tf.transpose(dot(W1, X), [1, 0, 2]), W2)
#print("W1XW2 shape {}".format(W1XW2.get_shape()))



