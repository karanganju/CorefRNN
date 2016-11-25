import tensorflow as tf
import numpy as np
import h5py as h5
import csv
from conllfeatures import getMentionFeats2
from conllfeatures import getPairFeats
import sys

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
WA_WIDTH = 128
WP_WIDTH = 128
FL_PENALTY = 0.5
FN_PENALTY = 1.2
WL_PENALTY = 1
LEARNING_RATE = 0.005
W2V_MIN_COUNT = 1
W2V_SIZE = 200
W2V_WINDOW = 5
ITERATION_COUNT = 5


# Get training and test data
cluster_data = getClustersArrayForMentions()
mentionFeats = getMentionFeats2("mentionsList.txt","wordsList.txt",W2V_MIN_COUNT,W2V_SIZE,W2V_WINDOW)

# Build Model for Local Mention Ranking


# Inputs/Placeholders (assuming we train one mention at a time)
# Here phia/p are the feature embeddings while Y is the best antecedent (or should we take cluster instead? - depends on output)
Phia_x = tf.placeholder(tf.float32, [1, PHIA_FEATURE_LEN])
Phip_x = tf.placeholder(tf.float32, [1, PHIP_FEATURE_LEN])
index_x = tf.placeholder(tf.int32, [1])

Phip_y = tf.placeholder(tf.float32, [1, PHIP_FEATURE_LEN])
index_y = tf.placeholder(tf.int32, [1])

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
l_p = tf.add(tf.matmul(Phip_x, W_p), tf.tile(b_p, [1, 1]))
l_p_concat = tf.concat(1, [l_a, l_p])

l_p_y = tf.add(tf.matmul(Phip_y, W_p), tf.tile(b_p, [1, 1]))
l_p_concat_y = tf.concat(1, [l_a, l_p_y])


# Fill best antecedent using max and all
f_x_ana = tf.add(tf.matmul(tf.nn.tanh(l_p_concat), u), tf.fill([1, 1], b_u[0]))
f_x_nonana = tf.add(tf.matmul(tf.nn.tanh(l_a), v), b_v)
f_x = tf.select(tf.equal(index_x, tf.constant(0, dtype='int32')), f_x_ana, f_x_nonana)

f_y_ana = tf.add(tf.matmul(tf.nn.tanh(l_p_concat_y), u), tf.fill([1, 1], b_u[0]))
f_y_nonana = tf.add(tf.matmul(tf.nn.tanh(l_a), v), b_v)
f_y = tf.select(tf.equal(index_y, tf.constant(0, dtype='int32')), f_y_ana, f_y_nonana)


loss_multiplier = tf.select(tf.equal(index_y, tf.constant(0, dtype='int32'))[0], tf.constant(FL_PENALTY, dtype='float32'), tf.select(tf.equal(index_x, tf.constant(0, dtype='int32'))[0],tf.constant(FN_PENALTY, dtype='float32'),tf.constant(WL_PENALTY, dtype='float32')))
loss_factor = tf.select(tf.equal(index_x,index_y)[0], tf.constant(0, dtype='float32'), loss_multiplier) 

loss = tf.mul(tf.add(tf.constant(1.0), tf.sub(f_x, f_y)), loss_factor)

train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

# Train model
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())

	# for all documents
	
	TRAINING_SIZE = 1000
	
	for iteration_count in range(ITERATION_COUNT):
		
		for i in range(TRAINING_SIZE):

			nonana_score = (np.array(sess.run(f_x_nonana, feed_dict={Phia_x: mentionFeats[i].reshape(1, PHIA_FEATURE_LEN)})))[0][0]

			best_index = 0
			best_score = nonana_score

			best_ant_index = -1
			best_ant_score = float('-inf')

			mentionFeats_i = mentionFeats[i].reshape(1, PHIP_FEATURE_LEN)

			for j in range(i):

				score = np.array(sess.run(f_x_ana, feed_dict={Phia_x: mentionFeats_i, Phip_x: getPairFeats(i, mentionFeats, PHIP_FEATURE_LEN)[j].reshape([1,PHIP_FEATURE_LEN])}))
				if (score >= best_score):
					best_score = score
					best_index = j+1

				if (cluster_data[i] == cluster_data[j] and score >= best_ant_score):
					best_ant_score = score
					best_ant_index = j+1
				
			if (best_ant_index == -1):
				best_ant_score = nonana_score
				best_ant_index = 0

			training_phip_x = np.zeros([1, PHIP_FEATURE_LEN])
			training_phip_y = np.zeros([1, PHIP_FEATURE_LEN])
			if (best_index != 0):
				training_phip_x = getPairFeats(i, mentionFeats ,PHIP_FEATURE_LEN)[best_index-1]
			if (best_ant_index != 0):
				training_phip_y = getPairFeats(i, mentionFeats ,PHIP_FEATURE_LEN)[best_ant_index-1]

			sess.run(train_op, feed_dict={Phia_x: mentionFeats_i, Phip_x: training_phip_x.reshape([1,PHIP_FEATURE_LEN]), Phip_y: training_phip_y.reshape([1,PHIP_FEATURE_LEN]), index_x: [best_index], index_y: [best_ant_index]})
			if (i % 200 == 0):
				print i


		print iteration_count, "th iteration done"

	score = 0

	for i in range(TRAINING_SIZE):

		best_index = 0
		best_score = nonana_score

		mentionFeats_i = mentionFeats[i].reshape(1, PHIP_FEATURE_LEN)

		for j in range(i):

			score = np.array(sess.run(f_x_ana, feed_dict={Phia_x: mentionFeats_i, Phip_x: getPairFeats(i, mentionFeats, PHIP_FEATURE_LEN)[j].reshape([1,PHIP_FEATURE_LEN])}))
			if (score >= best_score):
				best_score = score
				best_index = j+1

		print i+1, "mention is clustered with", best_index


		if (best_index == 0):
				score = score + 1
				for j in range(i):
					if (cluster_data[j] == cluster_data[i]):
						score = score - 1
						break
			elif (cluster_data[best_index-1] == cluster_data[i]):
				score = score + 1

			# print(i, sess.run(f_x, feed_dict={Phia_x: mentionFeats[i].reshape(1,W2V_SIZE) ,Phip_x: getPairFeats(i, mentionFeats, W2V_SIZE) ,Y_antecedent: latent_antecedents, mask: np.append([[1]],mask_arr).reshape([TRAINING_SIZE + 1,1])}))
			# print(i, sess.run(mask, feed_dict={Phia_x: np.random.rand(1, PHIA_FEATURE_LEN),Phip_x: np.random.rand(TRAINING_SIZE, PHIP_FEATURE_LEN),Y_antecedent: np.random.rand(1, TRAINING_SIZE + 1)}))
		print iteration_count, score, (score*100.0)/TRAINING_SIZE