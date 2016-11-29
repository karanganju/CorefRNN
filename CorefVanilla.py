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
LEARNING_RATE = 0.01
W2V_MIN_COUNT = 1
W2V_SIZE = 200
W2V_WINDOW = 5
ITERATION_COUNT = 1
TRAIN_DIR = "./Data/Train/"
TEST_DIR = "./Data/Test/"
CKPT_PATH = "./Checkpoints/local.ckpt"
RESTORE = False
SAVE = False

opts, args = getopt.getopt(sys.argv[1:],"n:l:d:rs:",[])
for opt, arg in opts:
	if opt == '-n':
		ITERATION_COUNT = int(arg)
	elif opt == '-l':
		LEARNING_RATE = float(arg)
	elif opt == '-d':
		TRAIN_DIR = arg + "/Train"
		TEST_DIR = arg + "/Test"
	elif opt == '-r':
		RESTORE = True
	elif opt == '-s':
		SAVE = True
		CKPT_PATH = arg

train_wordfiles = filter(lambda filename:  filename.endswith('wordsList.txt') , listdir(TRAIN_DIR))
test_wordfiles = filter(lambda filename:  filename.endswith('wordsList.txt') , listdir(TEST_DIR))
NUM_FILES = len(train_wordfiles)

# Build Model for Local Mention Ranking
# Inputs/Placeholders (assuming we train one mention at a time)
# Here phia/p are the feature embeddings while Y is the best antecedent (or should we take cluster instead? - depends on output)
Phia_x = tf.placeholder(tf.float32, [1, PHIA_FEATURE_LEN])
Phip_x = tf.placeholder(tf.float32, [None, PHIP_FEATURE_LEN])

# Y_antecedent array has True where it belongs to the same cluster and False otherwise
Y_antecedent = tf.placeholder(tf.float32, [None, 1])

tr_size = tf.shape(Phip_x)[0]

# Variables/Parameters
W_a = tf.Variable(tf.random_uniform([PHIA_FEATURE_LEN, WA_WIDTH]), name="W_a")
b_a = tf.Variable(tf.random_uniform([1, WA_WIDTH]), name="b_a") 
W_p = tf.Variable(tf.random_uniform([PHIP_FEATURE_LEN, WP_WIDTH]), name="W_p")
b_p = tf.Variable(tf.random_uniform([1, WP_WIDTH]), name="b_p")
u = tf.Variable(tf.random_uniform([WA_WIDTH + WP_WIDTH, 1]), name="u")
v = tf.Variable(tf.random_uniform([WA_WIDTH, 1]), name="v")
b_u = tf.Variable(tf.random_uniform([1]), name="b_u")
b_v = tf.Variable(tf.random_uniform([1]), name="b_v")

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

saver = tf.train.Saver()

# Train model
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	if (RESTORE == True):
		saver.restore(sess, CKPT_PATH)
	for iteration_count in range(ITERATION_COUNT):
		for train_doc in train_wordfiles:

			wordFile = TRAIN_DIR + train_doc
			mentionFile = wordFile.replace("wordsList", "mentionsList")
	
			try:
				cluster_data = getClustersArrayForMentions(mentionFile)
				mentionFeats = getMentionFeats2(mentionFile,wordFile,W2V_MIN_COUNT,W2V_SIZE,W2V_WINDOW)
			except:
				print train_doc
				continue

			TRAINING_SIZE = len(cluster_data)

			for i in range(TRAINING_SIZE):

				latent_antecedents = np.logical_not(cluster_data[:i] - cluster_data[i]).astype(np.int)
				latent_antecedents = np.append(np.array([not latent_antecedents.any()]).astype(np.int), latent_antecedents).reshape([i+1,1])

				sess.run(train_op, feed_dict={Phia_x: mentionFeats[i].reshape(1,W2V_SIZE), Phip_x: getPairFeats(i, mentionFeats, W2V_SIZE), Y_antecedent: latent_antecedents})
	
	eval_prec = 0
	eval_rec = 0
	total_ments = 0

	for test_doc in test_wordfiles:

		print test_doc
		
		wordFile = TEST_DIR + test_doc
		mentionFile = wordFile.replace("wordsList", "mentionsList")
			
		cluster_data = getClustersArrayForMentions(mentionFile)
		mentionFeats = getMentionFeats2(mentionFile,wordFile,W2V_MIN_COUNT,W2V_SIZE,W2V_WINDOW)

		TRAINING_SIZE = len(cluster_data)

		cluster_pred = np.zeros(TRAINING_SIZE)
		score = 0

		for i in range(TRAINING_SIZE):

			latent_antecedents = np.logical_not(cluster_data[:i] - cluster_data[i]).astype(np.int)
			latent_antecedents = np.append(np.array([not latent_antecedents.any()]).astype(np.int), latent_antecedents).reshape([i+1,1])

			cluster_pred[i] = np.array(sess.run(best_ant, feed_dict={Phia_x: mentionFeats[i].reshape(1,W2V_SIZE) ,Phip_x: getPairFeats(i, mentionFeats, W2V_SIZE) ,Y_antecedent: latent_antecedents}))
		
			# if (iteration_count == ITERATION_COUNT -1 and file_num == NUM_FILES - 1):
			# print i+1, cluster_pred[i]

			if (cluster_pred[i] == 0):
				score = score + 1
				for j in range(i):
					if (cluster_data[j] == cluster_data[i]):
						score = score - 1
						break
			elif (cluster_data[cluster_pred[i]-1] == cluster_data[i]):
				score = score + 1

		# print wordFile
		(_, rec, prec) = BCubedF1(cluster_data, cluster_pred)
		# print score, rec, prec, (score*100.0)/TRAINING_SIZE
		eval_rec += rec*TRAINING_SIZE
		eval_prec += prec*TRAINING_SIZE
		total_ments += TRAINING_SIZE

	print "Total weighted recall :", eval_rec/total_ments
	print "Total weighted precision :", eval_prec/total_ments
	print "B1 score :", (2 * eval_rec * eval_prec)/((eval_prec + eval_rec) * total_ments)

	if (SAVE == True):
		saver.save(sess, CKPT_PATH)
	print "OVER"