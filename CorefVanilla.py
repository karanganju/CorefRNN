import tensorflow as tf
import numpy as np
import h5py as h5
import csv
from conllfeatures import getMentionFeats
from conllfeatures import getPairFeats

def getClustersArrayForMentions():
	f = open('mentionsList.txt', 'r')
	csvreader = csv.reader(f, delimiter=' ')
	mylist = []
	for row in csvreader:
		mylist.append(int(row[1]))

	myarray = np.array(mylist)
	return myarray

# This is a test run generating only 
# o local mention ranking-based clusters (no global cluster evel features) 
# o using an end-to-end NN (no training on subtasks)
# o 2 layer model (g is the identity function)

# Config Variables
PHIA_FEATURE_LEN = 200
PHIP_FEATURE_LEN = 200
TRAINING_SIZE = 1253
WA_WIDTH = 128
WP_WIDTH = 128
FL_PENALTY = 0.5
FN_PENALTY = 1.2
WL_PENALTY = 1
LEARNING_RATE = 0.5
W2V_MIN_COUNT = 1
W2V_SIZE = 200
W2V_WINDOW = 5


# Get training and test data
# phip_tr_data = np.empty((TRAINING_SIZE, TRAINING_SIZE, PHIP_FEATURE_LEN))
# phia_tr_data = np.empty((TRAINING_SIZE, PHIA_FEATURE_LEN))
cluster_data = getClustersArrayForMentions()
mask_arr = np.zeros(TRAINING_SIZE)
mentionFeats = getMentionFeats("mentionsList.txt","wordsList.txt",W2V_MIN_COUNT,W2V_SIZE,W2V_WINDOW)

# Build Model for Local Mention Ranking


# Inputs/Placeholders (assuming we train one mention at a time)
# Here phia/p are the feature embeddings while Y is the best antecedent (or should we take cluster instead? - depends on output)
Phia_x = tf.placeholder(tf.float32, [1, PHIA_FEATURE_LEN])
Phip_x = tf.placeholder(tf.float32, [TRAINING_SIZE, PHIP_FEATURE_LEN])
# Y_antecedent array has True where it belongs to the same cluster and False otherwise
Y_antecedent = tf.placeholder(tf.float32, [1, TRAINING_SIZE + 1])

# Variables/Parameters
W_a = tf.Variable(tf.random_uniform([PHIA_FEATURE_LEN, WA_WIDTH]))
b_a = tf.Variable(tf.random_uniform([1, WA_WIDTH])) 
W_p = tf.Variable(tf.random_uniform([PHIP_FEATURE_LEN, WP_WIDTH]))
b_p = tf.Variable(tf.random_uniform([1, WP_WIDTH]))
u = tf.Variable(tf.random_uniform([WA_WIDTH + WP_WIDTH, 1]))
v = tf.Variable(tf.random_uniform([WA_WIDTH, 1]))
b_u = tf.Variable(tf.random_uniform([1]))
b_v = tf.Variable(tf.random_uniform([1]))

# Get inner linear function Wa(x)+ba and Wp(x)+bp
l_a = tf.add(tf.matmul(Phia_x, W_a),b_a)
l_a_tiled = tf.tile(l_a, [TRAINING_SIZE, 1])

l_p = tf.add(tf.matmul(Phip_x, W_p), tf.tile(b_p, [TRAINING_SIZE, 1]))
l_p_concat = tf.concat(1, [l_a_tiled, l_p])


# Fill best antecedent using max and all
f_x_ana = tf.add(tf.matmul(tf.nn.tanh(l_p_concat), u), tf.fill([TRAINING_SIZE, 1], b_u[0]))
f_x_nonana = tf.add(tf.matmul(tf.nn.tanh(l_a), v), b_v)

# Get argmax and max of ana and nonana f_x concatenated
f_x = tf.concat(0, [tf.fill([1,1], f_x_nonana[0][0]) ,f_x_ana])
best_ant = tf.argmax(f_x, 0)
f_x_best = tf.reduce_max(f_x, 0)

# Assign value to Y_antecedent somehow
f_x_reduced = tf.matmul(f_x, Y_antecedent)
f_y_latent = tf.reduce_max(f_x_reduced,0)
y_latent = tf.argmax(f_x_reduced,0)

loss_multiplier = tf.select(tf.equal(y_latent,tf.constant(0, dtype='int64'))[0], tf.constant(FL_PENALTY, dtype='float32'), tf.select(tf.equal(best_ant, tf.constant(0, dtype='int64'))[0],tf.constant(FN_PENALTY, dtype='float32'),tf.constant(WL_PENALTY, dtype='float32')))
loss_factor = tf.select(tf.equal(y_latent,best_ant)[0], tf.constant(0, dtype='float32'), loss_multiplier) 

loss = tf.mul(tf.add(tf.constant(1.0), tf.sub(f_x_best, f_y_latent)), loss_factor)
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

# Train model
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for i in range(TRAINING_SIZE):

		latent_antecedents = np.multiply(np.logical_not(cluster_data - cluster_data[i]).astype(np.int), mask_arr)
		latent_antecedents = np.append(np.array([not latent_antecedents.any()]).astype(np.int), latent_antecedents).reshape([1,TRAINING_SIZE + 1])

		sess.run(train_op, feed_dict={Phia_x: mentionFeats[i].reshape(1,W2V_SIZE) ,Phip_x: getPairFeats(i, mentionFeats, W2V_SIZE) ,Y_antecedent: latent_antecedents})	
		# print(sess.run(best_ant, feed_dict={Phia_x: np.random.rand(1, PHIA_FEATURE_LEN),Phip_x: np.random.rand(TRAINING_SIZE, PHIP_FEATURE_LEN),Y_antecedent: np.random.rand(1, TRAINING_SIZE + 1)}))

		mask_arr[i] = 1
		if (i % 10 == 0):
			print i

	mask_arr = np.zeros(TRAINING_SIZE)

	for i in range(TRAINING_SIZE):

		latent_antecedents = np.multiply(np.logical_not(cluster_data - cluster_data[i]).astype(np.int), mask_arr)
		latent_antecedents = np.append(np.array([not latent_antecedents.any()]).astype(np.int), latent_antecedents).reshape([1,TRAINING_SIZE + 1])

		print(i, sess.run(best_ant, feed_dict={Phia_x: mentionFeats[i].reshape(1,W2V_SIZE) ,Phip_x: getPairFeats(i, mentionFeats, W2V_SIZE) ,Y_antecedent: latent_antecedents}))
		# print(sess.run(best_ant, feed_dict={Phia_x: np.random.rand(1, PHIA_FEATURE_LEN),Phip_x: np.random.rand(TRAINING_SIZE, PHIP_FEATURE_LEN),Y_antecedent: np.random.rand(1, TRAINING_SIZE + 1)}))

		mask_arr[i] = 1
