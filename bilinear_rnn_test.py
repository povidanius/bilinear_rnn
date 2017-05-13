from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
from bilinear_rnn import *
import numpy as np
from helper import  *
import matplotlib.pyplot as plt

dx1 = 10
dx2 = 10
dh1 = 20 #20
dh2 = 20 #20
dy1 = 10
dy2 = 10
num_epochs = 1000
learning_rate = 0.002
batch_size = 128
n_steps = 20

def imshow(x):
	plt.imshow(x,cmap='gray')
	plt.show()

def imshow2(x,y):
	fig = plt.figure()
	ax = fig.add_subplot(2, 1, 1)
	ax.imshow(x,cmap='gray')
	ax.autoscale(False)
	ax2 = fig.add_subplot(2, 1, 2, sharex=ax, sharey=ax)
	ax2.imshow(y,cmap='gray')
	ax2.autoscale(False)
	ax.set_adjustable('box-forced')
	ax2.set_adjustable('box-forced')
	plt.show()

def normalize(x):
	return (x - mean(x, axis=0)) / std(A, axis=0)
	

def count_trainable_parameters():
 total_parameters = 0
 for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:      
        variable_parametes *= dim.value    
    total_parameters += variable_parametes
 return total_parameters

# "structure"
# first row - Gaussian
# remaining ones - multiplies of first row 
def get_sample():
	x = np.random.normal(loc = 0.0, scale = 1.0, size=(dx1,dx2))
	for i in range(dx1-1):
		x[i+1,:] = x[0,:] * np.random.normal(loc = 0.0, scale = 1.0)
	return x
# "no structure" - i.i.d. Gaussian entries.

def get_sample0():
	x = np.random.normal(loc = 0.0, scale = 1.0, size=(dx1,dx2))
	return x

def generate_data(num_samples = 1000):
	data_x = np.zeros((num_samples, n_steps, dx1, dx2))
	data_y = np.zeros((num_samples, dy1, dy2))

	for m in range(num_samples):

	  for n in range(n_steps):
	  	data_x[m, n,:,:] =  get_sample()

	  #data_y[m, :, :] = np.maximum(data_x[m,1,:,:], 0.5 * (data_x[m, 18,:,:] + data_x[m, 19,:,:]))
	  data_y[m, :, :] =  0.5 * (data_x[m,-15,:,:] + data_x[m, -1,:,:])
	return data_x, data_y
			
		
	
	
x = tf.placeholder("float", [None, n_steps, dx1, dx2])
y = tf.placeholder("float", [None, dy1, dy2])


weights = {
    'left': tf.Variable(tf.random_normal([dy1,dh1])),    
    'right': tf.Variable(tf.random_normal([dh2,dy2])),    
    'biases': tf.Variable(tf.random_normal([dy1, dy2]))
}



def RNN(x, weights):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input1, n_input2)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input1, n_input2)
	
   
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2, 3])
    # Reshaping to (n_steps*batch_size, n_input1*n_input2)
    x = tf.reshape(x, [-1, dx1*dx2])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input1*n_input2)
    x = tf.split(x, n_steps, 0)

    rnn_cell = BilinearGRU(input_shape = [dx1,dx2], hidden_shape = [dh1, dh2])
    #rnn_cell = rnn.GRUCell(dh1*dh2)

    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)



    out = tf.reshape(outputs[-1],[-1, dh1, dh2])

    # show!	
    prediction = dot(tf.transpose(dot(weights['left'], out), [1, 0, 2]), weights['right']) + weights['biases']
    #prediction = out

    return prediction


X_train,Y_train = generate_data(5000)
X_test, Y_test = generate_data(1000)
pred = RNN(x, weights)
loss = tf.reduce_mean(tf.square(tf.subtract(pred, y))) 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):

        start = 0
        end = batch_size
	batchloss = 0.0
        for i in range( int(1024/batch_size) ):

            batch_x = X_train[start:end,...]
            batch_y = Y_train[start:end,...]

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
	    batchloss = batchloss + sess.run(loss, feed_dict={x: batch_x, y: batch_y})

            start = end
            end = start + batch_size

	testloss = sess.run(loss, feed_dict={x: X_test, y: Y_test})
	print("Epoch {} train loss {}, testloss {}".format(epoch, batchloss / int(1024/batch_size), testloss))
	predictions = sess.run(pred, feed_dict={x: X_test})

imshow2(Y_test[1,...] , predictions[1,...])

print("Trainable parameters {}". format(count_trainable_parameters()))



