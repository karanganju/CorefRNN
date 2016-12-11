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

def MUCRecall(cluster_list, size_of_cluster):
	numerator = 0
	denominator = 0
	
	for elem in size_of_cluster:
		numerator += elem
		denominator += elem-1

	for List in cluster_list:
		present = np.zeros(len(size_of_cluster))

		for elem in List:
			present[elem] = 1

		numerator -= np.sum(present)

	if (numerator == 0):
		return 0
	return float(numerator)/float(denominator)

def MUCPrecision(cluster_list):
	numerator = 0
	denominator = 0
	
	for List in cluster_list:
		
		last = -1
		size = len(List)
		numerator += size
		denominator += size - 1

		for elem in List:
			if elem != last:
				numerator -= 1
				last = elem

	if (numerator == 0):
		return 0
	return float(numerator)/float(denominator)


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
def F1_scores(cluster_OPC, cluster_pred):

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
	if (recall+precision == 0):
		f1 = 0
	else:
		f1 = 200*recall*precision/(recall+precision)

	recall2 = MUCRecall(cluster_list, size_of_cluster)
	precision2 = MUCPrecision(cluster_list)
	if (recall2+precision2 == 0):
		f12 = 0
	else:
		f12 = 200*recall2*precision2/(recall2+precision2)

	return f1, 100*recall, 100*precision, f12, 100*recall2, 100*precision2

# a1 = [0,0,0,0,0,0,0,0,0,0,0,0]
# a2 = [0,0,0,0,0,1,1,2,2,2,2,2]
# F1_scores(a2,a1)