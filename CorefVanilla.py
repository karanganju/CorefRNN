import tensorflow as tf
import numpy as np

# This is a test run generating only 
# o local mention ranking-based clusters (no global cluster evel features) 
# o using an end-to-end NN (no training on subtasks)
# o 2 layer model (g is the identity function)

# We assume inputs are phia(x) and phip(x)

def read_data(array):
	return array

# Config Variables
PHIA_FEATURE_LEN = 10
PHIP_FEATURE_LEN = 10
TRAINING_SIZE = 100
WA_WIDTH = 128
WP_WIDTH = 700
FL_PENALTY = 0.1
FN_PENALTY = 0.1
WL_PENALTY = 0.1


# Get training and test data
phip_tr_data = numpy.empty((TRAINING_SIZE, TRAINING_SIZE, PHIP_FEATURE_LEN))
phia_tr_data = numpy.empty((TRAINING_SIZE, PHIA_FEATURE_LEN))


# Build Model for Local Mention Ranking

# Inputs/Placeholders
Phia_x = tf.placeholder(tf.float32, [TRAINING_SIZE, PHIA_FEATURE_LEN])
Phip_x = tf.placeholder(tf.float32, [TRAINING_SIZE, PHIP_FEATURE_LEN])
Y = tf.placeholder(tf.int32, [TRAINING_SIZE])

# Variables/Parameters
W_a = tf.Variable(tf.random_uniform([PHIA_FEATURE_LEN, WA_WIDTH]))
b_a = tf.Variable(tf.random_uniform([1, WA_WIDTH])) 
W_p = tf.Variable(tf.random_uniform([PHIP_FEATURE_LEN, WP_WIDTH]))
b_p = tf.Variable(tf.random_uniform([1, WP_WIDTH]))
u = tf.Variable(tf.random_uniform([WA_WIDTH + WP_WIDTH, 1]))
v = tf.Variable(tf.random_uniform([WA_WIDTH, 1]))
b_u = tf.Variable(tf.random_uniform([1,1]))
b_v = tf.Variable(tf.random_uniform([1,1]))

# Get inner linear function sorted out 
h_a = tf.nn.tanh(tf.matmul(Phia_x, W_a) + b_a)
h_p = tf.nn.tanh(tf.matmul(Phip_x, W_p) + b_p)

# Fill hbest using max and all
hbest = 0
loss = 


# Train model
with tf.Session() as sess:
