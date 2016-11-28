from gensim.models import Word2Vec
import csv
import numpy as np
import nltk
import random
from nltk.corpus import names

def gender_features(word):
    return {'suffix1': word[-1:], 'suffix2': word[-2:]}

def gender_classfier(self):
    labeled_names = (
        [(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
    random.shuffle(labeled_names)

    train_names = labeled_names[:]
    # test_names = labeled_names[:500]

    train_set = [(gender_features(n), gender) for (n, gender) in train_names]
    test_set = [(gender_features(n), gender) for (n, gender) in test_names]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    #print(classifier.classify(gender_features('Garima')))
    return classifier

def Word2VecModel(File, min_count = 1, size = 200, window = 5):
    file = open(File, "r")
    words = []
    sent = []
    words_tokenize = []
    for line in file:
        line = line.strip()
        line = line.replace("_"," ")
        if line == ".":
            words.append(sent)
            words_tokenize.append(line)
            sent = []
        else:
            sent.append(line)
            words_tokenize.append(line)
    file.close()

    if len(sent) != 0:
        words.append(sent)

    words_pos = nltk.pos_tag(words_tokenize)
    print(words_pos)

    model = Word2Vec(words, size=size, window=window, min_count=min_count)
    return model

def getMentionFeats(MentionFile, WordsFile, min_count, size, window):

    model = Word2VecModel(WordsFile, min_count, size, window)
    classifier = gender_classfier()

    file = open(MentionFile, "r")
    mentions = []
    mentionFeats = []
    words_tokenize = []
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

def Word2VecModel2(File, min_count = 1, size = 200, window = 5):
    file = open(File, "r")
    words = []
    sent = []
    words_tokenize = []
    for line in file:
        line = line.strip()
        line = line.replace("_"," ")
        line = line.split(" ")
        if line[0] == ".":
            words.append(sent)
            sent = []
            words_tokenize.append(line[0])
        else:
            for word in line:
                sent.append(word)
                words_tokenize.append(word)
    file.close()

    if len(sent) != 0:
        words.append(sent)

    model = Word2Vec(words, size=size, window=window, min_count=min_count)
    return (model,words_tokenize)

def getMentionFeats2(MentionFile, WordsFile, min_count, size, window):

    model = Word2VecModel2(WordsFile, min_count, size, window)

    file = open(MentionFile, "r")
    mentionFeats = []
    flag = 0
    for line in file:
        line = line.strip()
        line = line.split(" ")
        line = line[0]
        line = line.replace("_"," ")
        line = line.split(" ")
        total = np.zeros(size)
        flag = 0
        for word in line:
            total += model[word]
        total =  total/len(line)
        if flag == 1:
            mentionFeats = np.vstack((mentionFeats,total))
        else:
            mentionFeats.append(total)
            flag = 1
    file.close()
    return mentionFeats

def getPairFeats(idx,mentionFeats,size):
    PairwiseFeats = np.zeros((len(mentionFeats), size))
    for pidx in range(0,idx):
        feat1 = mentionFeats[idx]
        feat2 = mentionFeats[pidx]
        dist = feat1 - feat2
        PairwiseFeats[pidx] = dist
        #print(dist)
    return PairwiseFeats



if __name__ == '__main__':
    min_count = 1
    size = 200
    window = 5

    # mentionFeats = getMentionFeats("mentionsList.txt","wordsList.txt",min_count,size,window)

    # for idx in range(len(mentionFeats)):
    # print(getPairFeats(idx,mentionFeats,size))
