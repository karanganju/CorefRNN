from gensim.models import Word2Vec
import csv
import numpy as np

mentionFeats = []
PairwiseFeats = []

file = open("wordsList.txt", "r")
words = []
sent = []
for line in file:
    line = line.strip()
    line = line.replace("_"," ")
    if line == ".":
        words.append(sent)
        sent = []
    else:
        sent.append(line)
file.close()

if len(sent) != 0:
    words.append(sent)
#print(words)

min_count = 1
size = 200
window = 5
model = Word2Vec(words, size=size, window=window, min_count=min_count)

WordVocab = [k for (k, v) in model.vocab.items()]
#print(WordVocab)

file = open("mentionsList.txt", "r")
mentions = []
idx = 0
for line in file:
    line = line.strip()
    line = line.split(" ")
    line = line[0]
    line = line.replace("_"," ")
    mentions.append(line)
    mentionFeats.append((idx,model[line]))
file.close()

# for idx in range(len(mentionFeats)-1):
#     for pidx in range(idx+1,len(mentionFeats)):
#         feat1 = mentionFeats[idx][1]
#         feat2 = mentionFeats[pidx][1]+
#         dist = feat1 - feat2
#         PairwiseFeats.append((idx,pidx,dist))

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

