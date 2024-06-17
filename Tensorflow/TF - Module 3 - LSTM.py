# SIMPLE LSTM
import numpy as np
import tensorflow as tf
sess = tf.Session()

LSTM_CELL_SIZE = 4  # output size (dimension), which is same as hidden size in the cell

lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
state = (tf.zeros([1,LSTM_CELL_SIZE]),)*2
state

sample_input = tf.constant([[3,2,2,2,2,2]],dtype=tf.float32)
print (sess.run(sample_input))
with tf.variable_scope("LSTM_sample1"):
    output, state_new = lstm_cell(sample_input, state)
sess.run(tf.global_variables_initializer())
print (sess.run(state_new))
print (sess.run(output))
# [[3. 2. 2. 2. 2. 2.]]
# LSTMStateTuple(c=array([[-0.84185904, -0.01440321, -0.1277159 , -0.11322492]],
#       dtype=float32), h=array([[-0.4590366 , -0.00410132, -0.05832157, -0.039485  ]],
#       dtype=float32))
# [[-0.4590366  -0.00410132 -0.05832157 -0.039485  ]]
# h = previous output, c = previous state

# STACKED LSTM - 2 layer (output of 1st layer = input of 2nd layer)

import numpy as np
import tensorflow as tf
sess = tf.Session()
input_dim = 6
cells = []
# 1st layer
LSTM_CELL_SIZE_1 = 4 #4 hidden nodes
cell1 = tf.contrib.rnn.LSTMCell(LSTM_CELL_SIZE_1)
cells.append(cell1)
# 2nd layer
LSTM_CELL_SIZE_2 = 5 #5 hidden nodes
cell2 = tf.contrib.rnn.LSTMCell(LSTM_CELL_SIZE_2)
cells.append(cell2)

stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells) # multi-layer LSTM

#create RNN
# Batch size x time steps x features.
data = tf.placeholder(tf.float32, [None, None, input_dim])
output, state = tf.nn.dynamic_rnn(stacked_lstm, data, dtype=tf.float32)

#Batch size x time steps x features. (2,3,6)
sample_input = [[[1,2,3,4,3,2], [1,2,1,1,1,2],[1,2,2,2,2,2]],[[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]]
sample_input
output
sess.run(tf.global_variables_initializer())
sess.run(output, feed_dict={data: sample_input})
# array([[[ 0.05724197, -0.06358007,  0.04579141, -0.01807713,
#          -0.03480702],
#         [ 0.0963736 , -0.09612843,  0.09737915, -0.02968108,
#          -0.0453898 ],
#         [ 0.12735318, -0.13044626,  0.14718089, -0.03937736,
#          -0.06839415]],
#
#        [[ 0.05724197, -0.06358007,  0.04579141, -0.01807713,
#          -0.03480702],
#         [ 0.09595393, -0.0923373 ,  0.10166257, -0.03257579,
#          -0.0342318 ],
#         [ 0.11685655, -0.10286049,  0.13500528, -0.03076007,
#          -0.04047339]]], dtype=float32)
# output in the shape (2, 3, 5) corresponding to 2 batches, 3 states, 5 dimensions