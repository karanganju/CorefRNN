import tensorflow as tf
import numpy as np

# This is a test run generating only 
# o local mention ranking-based clusters (no global cluster evel features) 
# o using an end-to-end NN (no training on subtasks)
# o 2 layer model (g is the identity function)

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
phip_tr_data = np.empty((TRAINING_SIZE, TRAINING_SIZE, PHIP_FEATURE_LEN))
phia_tr_data = np.empty((TRAINING_SIZE, PHIA_FEATURE_LEN))


# Build Model for Local Mention Ranking


# Inputs/Placeholders (assuming we train one mention at a time)
Phia_x = tf.placeholder(tf.float32, [1, PHIA_FEATURE_LEN])
Phip_x = tf.placeholder(tf.float32, [TRAINING_SIZE, PHIP_FEATURE_LEN])
Y = tf.placeholder(tf.int32, [1])

# Variables/Parameters
W_a = tf.Variable(tf.random_uniform([PHIA_FEATURE_LEN, WA_WIDTH]))
b_a = tf.Variable(tf.random_uniform([1, WA_WIDTH])) 
W_p = tf.Variable(tf.random_uniform([PHIP_FEATURE_LEN, WP_WIDTH]))
b_p = tf.Variable(tf.random_uniform([1, WP_WIDTH]))
u = tf.Variable(tf.random_uniform([WA_WIDTH + WP_WIDTH, 1]))
v = tf.Variable(tf.random_uniform([WA_WIDTH, 1]))
b_u = tf.Variable(tf.random_uniform([1,1]))
b_v = tf.Variable(tf.random_uniform([1,1]))

# Get inner linear function Wa(x)+ba and Wp(x)+bp
l_a = tf.add(tf.matmul(Phia_x, W_a),b_a)
l_a_tiled = tf.tile(l_a, [1, TRAINING_SIZE])

l_p = tf.add(tf.matmul(Phip_x, W_p), tf.tile(b_p, [1, TRAINING_SIZE]))
l_p_concat = tf.concat(1, l_a_tiled, l_p)


# Fill best antecedent using max and all
f_x_ana = tf.matmul(tf.nn.tanh(l_p_concat), u) + tf.fill([TRAINING_SIZE, 1], b_u)
f_x_nonana = tf.add(tf.matmul(tf.nn.tanh(l_a), v), b_v)

# Get argmax and max of ana and nonana f_x concatenated
f_x = tf.concat(0, tf.fill([1,1], f_x_nonana) ,f_x_ana)
best_ant = tf.argmax(f_x, 0)
f_x_best = tf.reduce_max(f_x, 0)

# Get this somehow from Y
# f_y_latent = tf.

loss = 


# Train model
with tf.Session() as sess:


# Things to Do
# 1. Data Retrieval 
# 4. Use y to get latent antecedents - after we get data
# 5. Write out loss function - figure out
# 6. Write out training and run the session - after we get data
# 7. Buy beer for apartment

# Questions? 
# We are training one sample at a time right? No batches here?