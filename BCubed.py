import numpy as np

def BCubedRecall(cluster_list, size_of_cluster):
	rec_sum = 0
	num_mentions = 0

	for List in cluster_list:
		
		last = -1
		count = 0
		size = len(List)
		num_mentions += size

		for elem in List:
			if elem == last:
				count += 1
			
			else:
				rec_sum += float(count*count)/float(size_of_cluster[last])
				last = elem
				count = 1

		rec_sum += float(count*count)/float(size_of_cluster[last])
	return rec_sum/num_mentions


def BCubedPrecision(cluster_list):
	prec_sum = 0
	num_mentions = 0

	for List in cluster_list:
		
		last = -1
		count = 0
		size = len(List)
		num_mentions += size

		for elem in List:
			
			if elem == last:
				count += 1
			
			else:
				prec_sum += float(count*count)/float(size)
				last = elem
				count = 1

		prec_sum += float(count*count)/float(size)
	
	return prec_sum/num_mentions

	
# Assuming Cluster_OPC is indexed as array[i] = cluster of mention i starting from 0
# Assuming Cluster_pred is indexed as array[i] = antecedent of mention i (with 0 as dummy)
def BCubedF1(cluster_OPC, cluster_pred):

	num_pred_clusters = len(cluster_pred) - np.count_nonzero(cluster_pred)
	num_act_clusters = max(cluster_OPC) + 1

	size_of_cluster = np.zeros(num_act_clusters)
	cluster_list = [[] for i in range(num_pred_clusters)]
	cluster_num = np.zeros(len(cluster_pred), dtype=np.int)

	clusters_seen = -1
	for idx, val in enumerate(cluster_pred):
		if(val == 0):
			clusters_seen += 1
			cluster_num[idx] = clusters_seen
		else:
			cluster_num[idx] = cluster_num[val-1]

		size_of_cluster[cluster_OPC[idx]] += 1
		cluster_list[cluster_num[idx]].append(cluster_OPC[idx])

	for list_iter in cluster_list:
		list_iter.sort()

	recall = BCubedRecall(cluster_list, size_of_cluster)
	precision = BCubedPrecision(cluster_list)
	return 200*recall*precision/(recall+precision), 100*recall, 100*precision

# Test Case
# cluster_OPC = [0,0,0,0,0,1,1,2,1,3,4,1,1,1]
# cluster_pred = [0,1,1,1,0,5,5,0,8,8,8,8,8,8]

# a1 = [0,0,1,0,0,1,1,2,1,1]
# a2 = [2,1,3,3,2,1,0,0,1,3]
# print BCubedF1(a2,a1)