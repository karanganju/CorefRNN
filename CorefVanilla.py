import tensorflow as tf
import numpy as np
import sys
import getopt
from conllfeatures import *
from BCubed import BCubedF1
from os import listdir
from os.path import isfile, join

# Config Variables
PHIA_FEATURE_LEN = 200
PHIP_FEATURE_LEN = 200
WA_WIDTH = 128
WP_WIDTH = 128
FL_PENALTY = 0.5
FN_PENALTY = 1.2
WL_PENALTY = 1
LEARNING_RATE = 0.05
W2V_MIN_COUNT = 1
W2V_SIZE = 200
W2V_WINDOW = 5
ITERATION_COUNT = 1
DATA_DIR = "./Data/"
NUM_FILES = -1

opts, args = getopt.getopt(sys.argv[1:],"n:l:d:f:",[])
for opt, arg in opts:
	if opt == '-n':
		ITERATION_COUNT = int(arg)
	elif opt == '-l':
		LEARNING_RATE = float(arg)
	elif opt == '-d':
		DATA_DIR = arg
	elif opt == '-f':
		NUM_FILES = int(arg)

wordfiles = filter(lambda filename:  filename.endswith('wordsList.txt') , listdir(DATA_DIR))
if (NUM_FILES == -1):
	NUM_FILES = len(wordfiles)

# Build Model for Local Mention Ranking
# Inputs/Placeholders (assuming we train one mention at a time)
# Here phia/p are the feature embeddings while Y is the best antecedent (or should we take cluster instead? - depends on output)
Phia_x = tf.placeholder(tf.float32, [1, PHIA_FEATURE_LEN])
Phip_x = tf.placeholder(tf.float32, [None, PHIP_FEATURE_LEN])

# Y_antecedent array has True where it belongs to the same cluster and False otherwise
Y_antecedent = tf.placeholder(tf.float32, [None, 1])

tr_size = tf.shape(Phip_x)[0]

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
l_a_tiled = tf.tile(l_a, [tr_size, 1])

l_p = tf.add(tf.matmul(Phip_x, W_p), tf.tile(b_p, [tr_size, 1]))
l_p_concat = tf.concat(1, [l_a_tiled, l_p])

# Fill best antecedent using max and all
f_x_ana = tf.add(tf.matmul(tf.nn.tanh(l_p_concat), u), tf.fill([tr_size, 1], b_u[0]))
f_x_nonana = tf.add(tf.matmul(tf.nn.tanh(l_a), v), b_v)

# Get argmax and max of ana and nonana f_x concatenated
f_x = tf.add(tf.concat(0, [tf.fill([1,1], f_x_nonana[0][0]) ,f_x_ana]), tf.fill([tr_size + 1, 1], tf.constant(1000000, dtype='float32')))
best_ant = tf.argmax(f_x, 0)
f_x_best = tf.reduce_max(f_x, 0)

# Assign value to Y_antecedent somehow
f_x_reduced = tf.mul(f_x, Y_antecedent)
f_y_latent = tf.reduce_max(f_x_reduced,0)
y_latent = tf.argmax(f_x_reduced,0)

loss_multiplier = tf.select(tf.equal(y_latent,tf.constant(0, dtype='int64'))[0], tf.constant(FL_PENALTY, dtype='float32'), tf.select(tf.equal(best_ant, tf.constant(0, dtype='int64'))[0],tf.constant(FN_PENALTY, dtype='float32'),tf.constant(WL_PENALTY, dtype='float32')))
loss_factor = tf.select(tf.equal(y_latent,best_ant)[0], tf.constant(0, dtype='float32'), loss_multiplier) 

loss = tf.mul(tf.add(tf.constant(1.0), tf.sub(f_x_best, f_y_latent)), loss_factor)

train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

# Train model
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for file_num in range(NUM_FILES):

		wordFile = DATA_DIR + wordfiles[file_num]
		mentionFile = wordFile.replace("wordsList", "mentionsList")
		print wordFile
		
		cluster_data = getClustersArrayForMentions(mentionFile)
		mentionFeats = getMentionFeats2(mentionFile,wordFile,W2V_MIN_COUNT,W2V_SIZE,W2V_WINDOW)


		TRAINING_SIZE = len(cluster_data)

		for iteration_count in range(ITERATION_COUNT):
			for i in range(TRAINING_SIZE):

				latent_antecedents = np.logical_not(cluster_data[:i] - cluster_data[i]).astype(np.int)
				latent_antecedents = np.append(np.array([not latent_antecedents.any()]).astype(np.int), latent_antecedents).reshape([i+1,1])

				sess.run(train_op, feed_dict={Phia_x: mentionFeats[i].reshape(1,W2V_SIZE), Phip_x: getPairFeats(i, mentionFeats, W2V_SIZE), Y_antecedent: latent_antecedents})

			cluster_pred = np.zeros(TRAINING_SIZE)
			for i in range(TRAINING_SIZE):

				latent_antecedents = np.logical_not(cluster_data[:i] - cluster_data[i]).astype(np.int)
				latent_antecedents = np.append(np.array([not latent_antecedents.any()]).astype(np.int), latent_antecedents).reshape([i+1,1])

				# print(i+1, sess.run(loss_factor, feed_dict={Phia_x: mentionFeats[i].reshape(1,W2V_SIZE) ,Phip_x: getPairFeats(i, mentionFeats, W2V_SIZE) ,Y_antecedent: latent_antecedents, mask: np.append([[1]],mask_arr).reshape([TRAINING_SIZE + 1,1])}))
				# print(i+1, sess.run(loss, feed_dict={Phia_x: mentionFeats[i].reshape(1,W2V_SIZE) ,Phip_x: getPairFeats(i, mentionFeats, W2V_SIZE) ,Y_antecedent: latent_antecedents, mask: np.append([[1]],mask_arr).reshape([TRAINING_SIZE + 1,1])}))
				# print(i+1, sess.run(best_ant, feed_dict={Phia_x: mentionFeats[i].reshape(1,W2V_SIZE) ,Phip_x: getPairFeats(i, mentionFeats, W2V_SIZE) ,Y_antecedent: latent_antecedents, mask: np.append([[1]],mask_arr).reshape([TRAINING_SIZE + 1,1])}))
				
				cluster_pred[i] = np.array(sess.run(best_ant, feed_dict={Phia_x: mentionFeats[i].reshape(1,W2V_SIZE) ,Phip_x: getPairFeats(i, mentionFeats, W2V_SIZE) ,Y_antecedent: latent_antecedents}))
			
			print BCubedF1(cluster_pred, cluster_data)
			# 	if (iteration_count == ITERATION_COUNT -1):
			# 		print i+1, ant
			# 	if (ant == 0):
			# 		score = score + 1
			# 		for j in range(i):
			# 			if (cluster_data[j] == cluster_data[i]):
			# 				score = score - 1
			# 				break
			# 	elif (cluster_data[ant-1] == cluster_data[i]):
			# 		score = score + 1 

			# print iteration_count, score, (score*100.0)/TRAINING_SIZE