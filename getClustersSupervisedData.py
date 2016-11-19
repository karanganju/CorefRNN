import csv
import numpy as np
def getClustersArrayForMentions():
	f = open('mentionsList.txt', 'r')
	csvreader = csv.reader(f, delimiter=' ')
	mylist = []
	for row in csvreader:
		mylist.append(row[1])

	myarray = np.array(mylist)
	return myarray
#endDef

print(getClustersArrayForMentions())