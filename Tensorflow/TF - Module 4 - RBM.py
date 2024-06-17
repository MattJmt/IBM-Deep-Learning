# RBM is a Generative Model = specify a probability distribution over a dataset of input vectors
# In an unsupervised task, we try to form a model for P(x), where P is the probability given x as an input vector.
# In the supervised task, we first form a model for P(x|y), where P is the probability of x given y(the label for x).
# If we manage to find P(x|y) and P(y), then we can use Bayes rule to estimate P(y|x), because: 𝑝(𝑦|𝑥)=𝑝(𝑥|𝑦)𝑝(𝑦)𝑝(𝑥)

# 1) INITIALISATION
import urllib.request
with urllib.request.urlopen("http://deeplearning.net/tutorial/code/utils.py") as url:
    response = url.read()
target = open('utils.py', 'w')
target.write(response.decode('utf-8'))
target.close()
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#!pip install pillow
from PIL import Image
from utils import tile_raster_images
import matplotlib.pyplot as plt
%matplotlib inline

# 2) RBM LAYERS (has 2)
# 1st layer = visible/input layer. 2nd layer = hidden layer.
v_bias = tf.placeholder("float", [7]) # bias on 7 input nodes
h_bias = tf.placeholder("float", [2]) # bias on 2 hidden layer neurons
W = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(7, 2)).astype(np.float32)) # weight between 2 layers

# 3) RBM CAPABILITY AFTER LEARNING
# Phase 1: Forward pass - Input 1 training sample (image) X through all visible nodes, and pass it to all hidden nodes.
# Processing happens in each node in the hidden layer.
# This computation begins by making stochastic decisions about whether to transmit that input or not (i.e. to determine the state of each hidden layer).
# At the hidden layer's nodes, X is multiplied by a  𝑊𝑖𝑗  and added to h_bias.
# The result of those two operations is fed into the sigmoid function, which produces the node’s output,  𝑝(ℎ𝑗) , where j is the unit number.
# 𝑝(ℎ𝑗)=𝜎(∑𝑖𝑤𝑖𝑗𝑥𝑖) , where  𝜎()  is the logistic function.
# Now lets see what  𝑝(ℎ𝑗)  represents. In fact, it is the probabilities of the hidden units.And, all values together are called probability distribution.
# That is, RBM uses inputs x to make predictions about hidden node activations. For example, imagine that the values of  ℎ𝑝  for the first training item is [0.51 0.84].
# It tells you what is the conditional probability for each hidden neuron to be at Phase 1):
# p( ℎ1  = 1|V) = 0.51
# ( ℎ2  = 1|V) = 0.84
# As a result, for each row in the training set, a vector/tensor is generated, which in our case it is of size [1x2], and totally n vectors ( 𝑝(ℎ) =[nx2]).
# We then turn unit  ℎ𝑗  on with probability  𝑝(ℎ𝑗|𝑉) , and turn it off with probability  1−𝑝(ℎ𝑗|𝑉) .
# Therefore, the conditional probability of a configuration of h given v (for a training sample) is:
# 𝑝(𝐡∣𝐯)=∏𝑗=0𝐻𝑝(ℎ𝑗∣𝐯)
# Now, sample a hidden activation vector h from this probability distribution  𝑝(ℎ𝑗) . That is, we sample the activation vector from the probability distribution of hidden layer values.
# Assume that we have a trained RBM,an input vector such as [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]:
sess = tf.Session()
X = tf.constant([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
v_state = X
print ("Input: ", sess.run(v_state))

h_bias = tf.constant([0.1, 0.1])
print ("hb: ", sess.run(h_bias))
print ("w: ", sess.run(W))

# Calculate the probabilities of turning the hidden units on:
h_prob = tf.nn.sigmoid(tf.matmul(v_state, W) + h_bias)  #probabilities of the hidden units
print ("p(h|v): ", sess.run(h_prob))

# Draw samples from the distribution:
h_state = tf.nn.relu(tf.sign(h_prob - tf.random_uniform(tf.shape(h_prob)))) #states
print ("h0 states:", sess.run(h_state))

# Phase 2) Backward Pass (Reconstruction)- The RBM reconstructs data by making several forward and backward passes between the visible and hidden layers.
# So, in the second phase (i.e. reconstruction phase), the samples from the hidden layer (i.e. h) play the role of input.
# That is, h becomes the input in the backward pass. The same weight matrix and visible layer biases are used to go through the sigmoid function.
# The produced output is a reconstruction which is an approximation of the original input.
vb = tf.constant([0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1])
print ("b: ", sess.run(vb))
v_prob = sess.run(tf.nn.sigmoid(tf.matmul(h_state, tf.transpose(W)) + vb))
print ("p(vi∣h): ", v_prob)
v_state = tf.nn.relu(tf.sign(v_prob - tf.random_uniform(tf.shape(v_prob))))
print ("v probability states: ", sess.run(v_state))
# RBM learns a probability distribution over the input, and then, after being trained, the RBM can generate new samples from the learned probability distribution.
# As you know, probability distribution, is a mathematical function that provides the probabilities of occurrence of different possible outcomes in an experiment.
# The (conditional) probability distribution over the visible units v is given by:
# 𝑝(𝐯∣𝐡)=∏𝑉𝑖=0𝑝(𝑣𝑖∣𝐡),
# where, 𝑝(𝑣𝑖∣𝐡)=𝜎(𝑎𝑖+∑𝐻𝑗=0𝑤𝑗𝑖ℎ𝑗)
# so, given current state of hidden units and weights, what is the probability of generating [1. 0. 0. 1. 0. 0. 0.] in reconstruction phase, based on the above probability distribution function?
inp = sess.run(X)
print(inp)
print(v_prob[0])
v_probability = 1
for elm, p in zip(inp[0],v_prob[0]) :
    if elm ==1:
        v_probability *= p
    else:
        v_probability *= (1-p)
v_probability

# Code yields:
# Input:  [[1. 0. 0. 1. 0. 0. 0.]]
# hb:  [0.1 0.1]
# w:  [[ 0.21674757 -0.9082403 ]
#  [-0.24306789  0.04334203]
#  [ 0.3776022  -0.25756997]
#  [ 1.1377761  -0.43161273]
#  [-0.0967419  -0.55682766]
#  [ 0.8154882   0.05079272]
#  [ 0.01717485 -1.7094853 ]]
# p(h|v):  [[0.8106937  0.22446156]]
# h0 states: [[0. 0.]]
# b:  [0.1 0.2 0.1 0.1 0.1 0.2 0.1]
# p(vi∣h):  [[0.35629243 0.50006855 0.5547872  0.6912914  0.36503667 0.74388903
#   0.16905908]]
# v probability states:  [[1. 1. 1. 1. 0. 1. 0.]]
# [[1. 0. 0. 1. 0. 0. 0.]]
# [0.35629243 0.50006855 0.5547872  0.6912914  0.36503667 0.74388903
#  0.16905908]
# 0.0074078604635678235

# MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX[1].shape
vb = tf.placeholder("float", [784]) # 784 pixels
hb = tf.placeholder("float", [50]) # 50 nodes in hidden layer
W = tf.placeholder("float", [784, 50]) # tensor representing weights between neurons
v0_state = tf.placeholder("float", [None, 784]) # A visible layer
h0_prob = tf.nn.sigmoid(tf.matmul(v0_state, W) + hb)  # probabilities of the hidden units
h0_state = tf.nn.relu(tf.sign(h0_prob - tf.random_uniform(tf.shape(h0_prob)))) #sample_h_given_X
v1_prob = tf.nn.sigmoid(tf.matmul(h0_state, tf.transpose(W)) + vb) # define reconstruction part
v1_state = tf.nn.relu(tf.sign(v1_prob - tf.random_uniform(tf.shape(v1_prob)))) #sample_v_give
# objective of function: Maximize the likelihood of our data being drawn from that distribution
err = tf.reduce_mean(tf.square(v0_state - v1_state))

# 4) TRAINING THE MODEL
# using the gradient descent is hard (due to complex maths where derivative needs to be calculated) so use different approach:
# a) Gibbs Sampling
# First, given an input vector v we are using p(h|v) for prediction of the hidden values h.
# - 𝑝(ℎ|𝑣)=𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝑋⊗𝑊+ℎ𝑏)
# - h0 = sampleProb(h0)
# Then, knowing the hidden values, we use p(v|h) for reconstructing of new input values v.
# - 𝑝(𝑣|ℎ)=𝑠𝑖𝑔𝑚𝑜𝑖𝑑(ℎ0⊗𝑡𝑟𝑎𝑛𝑠𝑝𝑜𝑠𝑒(𝑊)+𝑣𝑏)
# - 𝑣1=𝑠𝑎𝑚𝑝𝑙𝑒𝑃𝑟𝑜𝑏(𝑣1)  (Sample v given h)
# This process is repeated k times. After k iterations we obtain an other input vector vk which was recreated from original input values v0 or X.
# Reconstruction steps:
# Get one data point from data set, like x, and pass it through the net
# Pass 0: (x)  ⇒  (h0)  ⇒  (v1) (v1 is reconstruction of the first pass)
# Pass 1: (v1)  ⇒  (h1)  ⇒  (v2) (v2 is reconstruction of the second pass)
# Pass 2: (v2)  ⇒  (h2)  ⇒  (v3) (v3 is reconstruction of the third pass)
# Pass n: (vk)  ⇒  (hk+1)  ⇒  (vk+1)(vk is reconstruction of the nth pass)
# b) contrastive divergence (CD-k)
# The update of the weight matrix is done during the Contrastive Divergence step.
# Vectors v0 and vk are used to calculate the activation probabilities for hidden values h0 and hk. The difference between the outer products of those probabilities with input vectors v0 and vk results in the update matrix:
# Δ𝑊=𝑣0*ℎ0−𝑣𝑘*ℎ𝑘
# Contrastive Divergence is actually matrix of values that is computed and used to adjust values of the W matrix. Changing W incrementally leads to training of W values. Then on each step (epoch), W is updated to a new value W' through the equation below:
# 𝑊′=𝑊+𝑎𝑙𝑝ℎ𝑎∗Δ𝑊
# Alpha = "learning rate".
# Assume that k=1:
h1_prob = tf.nn.sigmoid(tf.matmul(v1_state, W) + hb)
h1_state = tf.nn.relu(tf.sign(h1_prob - tf.random_uniform(tf.shape(h1_prob)))) #sample_h_given_X
alpha = 0.01
W_Delta = tf.matmul(tf.transpose(v0_state), h0_prob) - tf.matmul(tf.transpose(v1_state), h1_prob)
update_w = W + alpha * W_Delta
update_vb = vb + alpha * tf.reduce_mean(v0_state - v1_state, 0)
update_hb = hb + alpha * tf.reduce_mean(h0_state - h1_state, 0)
# initialise variables and start a session:
cur_w = np.zeros([784, 50], np.float32)
cur_vb = np.zeros([784], np.float32)
cur_hb = np.zeros([50], np.float32)
prv_w = np.zeros([784, 50], np.float32)
prv_vb = np.zeros([784], np.float32)
prv_hb = np.zeros([50], np.float32)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# error in the first run:
sess.run(err, feed_dict={v0_state: trX, W: prv_w, vb: prv_vb, hb: prv_hb})
#Parameters
epochs = 5
batchsize = 100
weights = []
errors = []

for epoch in range(epochs):
    for start, end in zip( range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        cur_w = sess.run(update_w, feed_dict={ v0_state: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0_state: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={ v0_state: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
        if start % 10000 == 0:
            errors.append(sess.run(err, feed_dict={v0_state: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
            weights.append(cur_w)
    print ('Epoch: %d' % epoch,'reconstruction error: %f' % errors[-1])
plt.plot(errors)
plt.xlabel("Batch Number")
plt.ylabel("Error")
plt.show()
# final weight after training
uw = weights[-1].T
print (uw) # a weight matrix of shape (50,784)
# Code yields:
# Epoch: 0 reconstruction error: 0.100013
# Epoch: 1 reconstruction error: 0.095585
# Epoch: 2 reconstruction error: 0.093358
# Epoch: 3 reconstruction error: 0.092155
# Epoch: 4 reconstruction error: 0.091577
# graph of error
# [[-0.33603925 -0.29942685 -0.30367956 ... -0.29058355 -0.2751093
#   -0.24873108]
#  [-0.47675952 -0.4008309  -0.43337438 ... -0.39269093 -0.4016417
#   -0.44480753]
#  [-0.27123988 -0.27836284 -0.25084674 ... -0.2988852  -0.2910491
#   -0.25535417]
#  ...
#  [-0.745069   -0.74891686 -0.8425641  ... -0.95763427 -0.80271995
#   -0.859978  ]
#  [-0.77272165 -0.68676025 -0.7508897  ... -0.75758374 -0.71844786
#   -0.7358197 ]
#  [-0.38586885 -0.35702246 -0.34271437 ... -0.35294366 -0.34629837
#   -0.36877722]]




# 5) LEARNED FEATURES
# visualize the connections between that hidden unit and each element in the input vector
# tile_raster_images helps in generating an easy to grasp image from a set of samples or weights.
# It transform the uw (with one flattened image per row of size 784), into an array (of size  25×20 ) in which images are reshaped and laid out like tiles on a floor.
tile_raster_images(X=cur_w.T, img_shape=(28, 28), tile_shape=(5, 10), tile_spacing=(1, 1))
import matplotlib.pyplot as plt
from PIL import Image
%matplotlib inline
image = Image.fromarray(tile_raster_images(X=cur_w.T, img_shape=(28, 28) ,tile_shape=(5, 10), tile_spacing=(1, 1)))
### Plot image
plt.rcParams['figure.figsize'] = (18.0, 18.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')
# Each tile in the above visualization corresponds to a vector of connections between a hidden unit and visible layer's units
# the gray color represents weight = 0, and the whiter it is, the more positive the weights are (closer to 1). Conversely, the darker pixels are, the more negative the weights.
# The positive pixels will increase the probability of activation in hidden units (after multiplying by input/visible pixels), and negative pixels will decrease the probability of a unit hidden to be 1 (activated).
from PIL import Image
image = Image.fromarray(tile_raster_images(X =cur_w.T[10:11], img_shape=(28, 28),tile_shape=(1, 1), tile_spacing=(1, 1)))
### Plot image
plt.rcParams['figure.figsize'] = (4.0, 4.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')
# the reconstruction of an image: plot the image
!wget -O destructed3.jpg  https://ibm.box.com/shared/static/vvm1b63uvuxq88vbw9znpwu5ol380mco.jpg
img = Image.open('destructed3.jpg')
img
# pass image through net
# convert the image to a 1d numpy array
sample_case = np.array(img.convert('I').resize((28,28))).ravel().reshape((1, -1))/255.0
# Feed the sample case into the network and reconstruct the output:
hh0_p = tf.nn.sigmoid(tf.matmul(v0_state, W) + hb)
#hh0_s = tf.nn.relu(tf.sign(hh0_p - tf.random_uniform(tf.shape(hh0_p))))
hh0_s = tf.round(hh0_p)
hh0_p_val,hh0_s_val  = sess.run((hh0_p, hh0_s), feed_dict={ v0_state: sample_case, W: prv_w, hb: prv_hb})
print("Probability nodes in hidden layer:" ,hh0_p_val)
print("activated nodes in hidden layer:" ,hh0_s_val)
# Code yields:
# Probability nodes in hidden layer: [[7.7247575e-10 5.6583758e-08 2.0225342e-34 7.3385922e-11 3.3130041e-18
#   9.2779917e-09 3.0394042e-12 1.4289566e-31 4.1205531e-18 1.0000000e+00
#   2.7689712e-05 4.0843058e-12 1.6541340e-36 2.2961540e-27 8.3293974e-26
#   9.9999988e-01 6.2953331e-09 1.2077861e-08 1.3728146e-13 1.9571030e-33
#   2.7959011e-03 2.9528591e-10 0.0000000e+00 1.3366980e-24 8.6360173e-29
#   1.0000000e+00 1.4567272e-26 3.4109786e-07 1.2264399e-10 9.9999940e-01
#   2.9622329e-11 3.5855426e-06 0.0000000e+00 1.7007238e-01 3.5699223e-32
#   8.9214974e-10 4.8945689e-13 7.8934573e-19 0.0000000e+00 4.4228480e-34
#   9.3008825e-13 5.9524094e-08 1.0000000e+00 9.9741793e-29 1.1295310e-24
#   3.4442415e-19 9.5023192e-14 2.0630519e-29 2.3138245e-22 4.5418608e-10]]
# activated nodes in hidden layer: [[0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.
#   0. 0.]]

# reconstruct
vv1_p = tf.nn.sigmoid(tf.matmul(hh0_s_val, tf.transpose(W)) + vb)
rec_prob = sess.run(vv1_p, feed_dict={ hh0_s: hh0_s_val, W: prv_w, vb: prv_vb})
# plot the reconstructed image:
img = Image.fromarray(tile_raster_images(X=rec_prob, img_shape=(28, 28),tile_shape=(1, 1), tile_spacing=(1, 1)))
plt.rcParams['figure.figsize'] = (4.0, 4.0)
imgplot = plt.imshow(img)
imgplot.set_cmap('gray')
# image of 3 was plotted, a bit blurry

