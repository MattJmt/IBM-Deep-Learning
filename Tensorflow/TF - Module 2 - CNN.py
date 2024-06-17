# Must-know functions
sess = tf.InteractiveSession() #create interactive session
x  = tf.placeholder(tf.float32, shape=[None, 784]) # placeholder X = 'space' allocated input (image)
# Each input has 784 pixels distributed in a 28*28 matrix
# shape = tensor size by dimensions
# 1st dimension = none : indicates batch(group of data) size can be any size
# 2nd dimension = 784. Indicates the number of pixels on a single flattened MNIST image.
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # placeholder Y = final output or labels
# 10 possible classes (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
# The 'shape' argument defines the tensor size by its dimensions.
# 1st dimension = None. Indicates that the batch size, can be of any size.
# 2nd dimension = 10. Indicates the number of targets/outcomes
# tf.float32 is a dtype = layout style (ex: integer, float)

W = tf.Variable(tf.zeros([784, 10],tf.float32)) # Weight tensor
b = tf.Variable(tf.zeros([10],tf.float32)) # Bias tensor
# run the op initialize_all_variables using an interactive session
sess.run(tf.global_variables_initializer()) # initialise weights and biases with null values
tf.matmul(x,W) + b # mathematical operation to add weights and biases to the inputs
y = tf.nn.softmax(tf.matmul(x,W) + b) # Softmax regression
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) # function that is used to minimize the difference between the right answers (labels) and estimated outputs by our Networ
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # gradient descent
#Load 50 training examples for each training iteration (mini-batch training)
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#Test
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print("The final accuracy for the simple ANN model is: {} % ".format(acc) )
sess.close() #finish the session
# HOW TO IMPROVE MODEL?
# Regularization of Neural Networks using DropConnect
# Multi-column Deep Neural Networks for Image Classification
# APAC: Augmented Pattern Classification with Neural Networks
# Simple Deep Neural Network with Dropout (more than 1 layer)

# Architecture of our network is:
# (Input) -> [batch_size, 28, 28, 1] >> Apply 32 filter of [5x5]
# (Convolutional layer 1) -> [batch_size, 28, 28, 32]
# (ReLU 1) -> [?, 28, 28, 32]
# (Max pooling 1) -> [?, 14, 14, 32]
# (Convolutional layer 2) -> [?, 14, 14, 64]
# (ReLU 2) -> [?, 14, 14, 64]
# (Max pooling 2) -> [?, 7, 7, 64]
# [fully connected layer 3] -> [1x1024]
# [ReLU 3] -> [1x1024]
# [Drop out] -> [1x1024]
# [fully connected layer 4] -> [1x10]

import tensorflow as tf

# finish possible remaining session
sess.close()

#Start interactive session
sess = tf.InteractiveSession()

# INPUT DATA (MNIST)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# initial parameters
width = 28 # width of the image in pixels
height = 28 # height of the image in pixels
flat = width * height # number of pixels in one image
class_output = 10 # number of possible classifications for the problem
# Input and output (placeholders)
x  = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])
# Convert images of data set to tensors
x_image = tf.reshape(x, [-1,28,28,1]) # 1st dimension: batch, 2nd: width, 3rd: height, 4th: image channels
x_image

# CONVOLUTIONAL LAYER 1
# define kernel
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)) # [filter_height, filter_width, in_channels, out_channels] // 32 = different filters applied
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs
# convolutional layer
convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1 # strides = shift, 2D = 2 dimensions
# ReLu
h_conv1 = tf.nn.relu(convolve1)
# maxpooling
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2_kernel and strides so no overlap
conv1 # output of first layer is 32 matrices of [14*14]

# CONVOLUTIONAL LAYER 2
# weights and biases of kernel
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)) # 64 filters of size [5*5*32]
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs
# convolve image with tensor and add biases
convolve2= tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
# ReLU
h_conv2 = tf.nn.relu(convolve2)
# Max pooling
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
conv2 # output of second layer is 64 matrices of [7*7]

# FULLY CONNECTED LAYER
# flattening 2nd layer
layer2_matrix = tf.reshape(conv2, [-1, 7 * 7 * 64])
# Weights and biases between layer 2 & 3
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1)) # 7*7*64 = 1027 = 1024 outpust to sotfmax layer
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs
# matrix multiplication
fcl = tf.matmul(layer2_matrix, W_fc1) + b_fc1
# ReLU
h_fc1 = tf.nn.relu(fcl)
h_fc1

# DROPOUT LAYER, optional phase for reducing overfitting
keep_prob = tf.placeholder(tf.float32) # some units randomly switched off to reduce amount of data
layer_drop = tf.nn.dropout(h_fc1, keep_prob)
layer_drop

# READOUT LAYER
# Weights and Biases
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]
# Matrix Multiplication
fc=tf.matmul(layer_drop, W_fc2) + b_fc2
# Apply the Softmax activation Function
y_CNN= tf.nn.softmax(fc)
y_CNN

# RECAP OF THE STRUCTURE:
# 0) Input - MNIST dataset
# 1) Convolutional and Max-Pooling
# 2) Convolutional and Max-Pooling
# 3) Fully Connected Layer
# 4) Processing - Dropout
# 5) Readout layer - Fully Connected
# 6) Outputs - Classified digits¶

# DEFINE FUNCTIONS AND TRAIN THE MODEL
# Define the loss function
import numpy as np
layer4_test =[[0.9, 0.1, 0.1],[0.9, 0.1, 0.1]]
y_test=[[1.0, 0.0, 0.0],[1.0, 0.0, 0.0]]
np.mean( -np.sum(y_test * np.log(layer4_test),1))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))
# define the optimiiser
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# Define prediction
correct_prediction = tf.equal(tf.argmax(y_CNN, 1), tf.argmax(y_, 1))
# Define accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Run session, train
sess.run(tf.global_variables_initializer())
for i in range(1100):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, float(train_accuracy)))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# EVALUATE THE MODEL
# evaluate in batches to avoid out-of-memory issues
n_batches = mnist.test.images.shape[0] // 50
cumulative_accuracy = 0.0
for index in range(n_batches):
    batch = mnist.test.next_batch(50)
    cumulative_accuracy += accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
print("test accuracy {}".format(cumulative_accuracy / n_batches))
# Visualisation (looking at filters)
kernels = sess.run(tf.reshape(tf.transpose(W_conv1, perm=[2, 3, 0,1]),[32, -1]))
!wget --output-document utils1.py http://deeplearning.net/tutorial/code/utils.py
import utils1
from utils1 import tile_raster_images
import matplotlib.pyplot as plt
from PIL import Image
%matplotlib inline
image = Image.fromarray(tile_raster_images(kernels, img_shape=(5, 5) ,tile_shape=(4, 8), tile_spacing=(1, 1)))
### Plot image
plt.rcParams['figure.figsize'] = (18.0, 18.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')
# output of image passing through 1st layer
import numpy as np
plt.rcParams['figure.figsize'] = (5.0, 5.0)
sampleimage = mnist.test.images[1]
plt.imshow(np.reshape(sampleimage,[28,28]), cmap="gray")
ActivatedUnits = sess.run(convolve1,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 6
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")
# output image 2nd layer
ActivatedUnits = sess.run(convolve2,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 8
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")
sess.close() #finish the session

# Extracting MNIST_data/train-images-idx3-ubyte.gz
# Extracting MNIST_data/train-labels-idx1-ubyte.gz
# Extracting MNIST_data/t10k-images-idx3-ubyte.gz
# Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
# step 0, training accuracy 0.14
# step 100, training accuracy 0.92
# step 200, training accuracy 0.9
# step 300, training accuracy 0.96
# step 400, training accuracy 1
# step 500, training accuracy 0.94
# step 600, training accuracy 0.94
# step 700, training accuracy 0.98
# step 800, training accuracy 0.94
# step 900, training accuracy 0.98
# step 1000, training accuracy 0.94
# test accuracy 0.966499999165535
