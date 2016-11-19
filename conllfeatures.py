from gensim.models import Word2Vec
import csv
import numpy as np

def Word2VecModel(File, min_count = 1, size = 200, window = 5):
    file = open(File, "r")
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

    model = Word2Vec(words, size=size, window=window, min_count=min_count)
    return model

def getMentionFeats(MentionFile, model):
    file = open(MentionFile, "r")
    mentions = []
    mentionFeats = []
    flag = 0
    for line in file:
        line = line.strip()
        line = line.split(" ")
        line = line[0]
        line = line.replace("_"," ")
        mentions.append(line)
        if flag == 1:
            mentionFeats = np.vstack((mentionFeats,model[line]))
        else:
            mentionFeats.append(model[line])
            flag = 1
    file.close()
    return mentionFeats

def getPairFeats(idx,mentionFeats):
    PairwiseFeats = np.zeros((len(mentionFeats), size))
    for pidx in range(0,idx):
        feat1 = mentionFeats[idx][1]
        feat2 = mentionFeats[pidx][1]
        dist = feat1 - feat2
        PairwiseFeats[pidx] = dist
    return PairwiseFeats


if __name__ == '__main__':
    min_count = 1
    size = 200
    window = 5

    model = Word2VecModel("wordsList.txt",min_count,size,window)
    mentionFeats = getMentionFeats("mentionsList.txt",model)

    for idx in range(len(mentionFeats)):
        PairwiseFeats = getPairFeats(idx,mentionFeats)