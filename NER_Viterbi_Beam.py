
from collections import Counter
import itertools
from sklearn.metrics import f1_score
import pandas as pd
import argparse
import time
import numpy as np
import random

# parse commandline arguments
parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('-v', action = 'store_true')
parser.add_argument('-b', action = 'store_true')
parser.add_argument("train")
parser.add_argument("test")

args= parser.parse_args()
viterbi = args.v
beam = args.b
train_path = args.train
test_path = args.test

def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None): 
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()] 
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()] 
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)


train = load_dataset_sents(train_path)
test = load_dataset_sents(test_path)
tagList = ['O','ORG','MISC','PER','LOC']

def cwcl(train):
    flat_train = [pair for sublist in train for pair in sublist]
    train_dict = dict(Counter(flat_train))
    return train_dict


def phi_1(sentence):
    phi=dict(Counter(sentence))
    return phi

def multiply_dict(phi, w):
    prod = 0
    for k in phi:
        if k in w:
            prod += w[k] * phi[k]
    return prod

def add(phi, w, coe=1):
    for k in phi:
        if k in w:
            w[k]=w[k]+coe*phi[k]
    return w

def subtract(phi, w, coe=1):
    for k in phi:
        if k in w:
            w[k]=w[k]-coe*phi[k]
    return w


# create an empty viterbi matrix whose dimension = |Y|*n
def viterbi_matrix(sentence):
    width = len (sentence)
    height = len(tagList)
    return np.zeros((height, width))


def train_viterbi(train, epoches=6):
    w = { k:0 for k in cwcl(train)}
    a = w.copy()
    step = epoches * len(train)
    # shuffle data at the beginning of each epoch
    for epoch in range(epoches):
        correct=0
        random.seed(788)
        random.shuffle(train)
        for sentence in train:
            V = viterbi_matrix(sentence)
            # first column of Viterbi matrix, that's the weight of (first word, tag)
            for i in range(len(tagList)):
                if (sentence[0][0], tagList[i]) in w:
                    V[i,0] = w[(sentence[0][0], tagList[i])]
                else :
                    V[i,0] = 0
            for i in range(1, len(sentence)):
                for j in range(len(tagList)):
                    # scoreArray includes scores of 5 paths toward one single node 
                    scoreArray=[w[(sentence[i][0],tagList[k])] if (sentence[i][0],tagList[k]) in w else 0 for k in range(5)]
                    V[j, i] = max(V[:,i-1]+scoreArray[j])
            # because the argmax values of each column are same, use a vector rather than matrix to represent pointers
            B = np.argmax(V,axis=0)
            # use back pointers to locate the prediction
            predSeq=[tagList[i] for i in B]
            if predSeq == [pair[1] for pair in sentence]:
                correct+=1
                #  do update
            else:
                w=add(phi_1(sentence),w)
                w=subtract(dict(Counter(predSeq)),w)
                a=add(phi_1(sentence),a, coe=step/(epoches*len(train)))
                a=subtract(dict(Counter(predSeq)),a,coe=step/(epoches*len(train)))
            step-=1
        print('incorrect ', len(train)-correct)
    return a


# same as a part of training function
def predict_viterbi(test, w):
    predictions=[]
    for sentence in test:
        V = viterbi_matrix(sentence)
        for i in range(len(tagList)):
            if (sentence[0][0], tagList[i]) in w:
                V[i,0] = w[(sentence[0][0], tagList[i])]
            else :
                V[i,0] = 0
        for i in range(1, len(sentence)):
            for j in range(len(tagList)):
                scoreArray=[w[(sentence[i][0],tagList[k])] if (sentence[i][0],tagList[k]) in w else 0 for k in range(5)]
                V[j, i] = max(V[:,i-1]+scoreArray[j])
        B = np.argmax(V,axis=0)
        predSeq=[tagList[i] for i in B]
        predictions.append(predSeq)
    return predictions

# evaluate predictions for both methods
def evaluation(test, w):
    if mode == 'beam':
        predictions = predict_beam(test, w)
    elif mode == 'viterbi':
        predictions = predict_viterbi(test, w)
    testSeq = [pair[1] for sentence in test for pair in sentence]
    predictSeq = [word for sentence in predictions for word in sentence]
    f1=f1_score(testSeq, predictSeq, average='micro', labels=['ORG','MISC', 'PER', 'LOC']) 
    print(mode+' f1 score:',f1)
    return f1
# beam search training, beam_size is a global variable which is defined outside the function.
def train_beam(train, epochs):
    w = { k:0 for k in cwcl(train)}
    a = w.copy()
    step = epochs * len(train)
    for epoch in range(epochs):
        correct=0
        random.seed(788)
        random.shuffle(train)
        for sentence in train:
            # initialize beam dictionary as {first word's tags: scores(weights)}
            startList= [(sentence[0][0], tagList[i]) for i in range(len(tagList))]
            B={ i:(w[i] if i in w else 0) for i in startList}
            # counter get the top n scored paths
            B=Counter(B).most_common(beam_size)
            B = {k[0][1]:k[1] for k in B}
            count = 0
            # start go forward
            for pair in sentence[1:]:
                count += 1
                B_prime={}
                for b in B:
                    for y in tagList:
                        wordSeq = [sentence[i][0] for i in range(count+1)]
                        # '_' connects tags. A path looks like 'O_ORG_PER_O'
                        tagSeq = b.split('_')+[y]
                        # calculate a path's score by combine it with current sentence's words
                        score = multiply_dict(Counter([(wordSeq[i], tagSeq[i]) for i in range(count+1)]),w)
                        B_prime['_'.join(tagSeq)] = score
                # keep top N
                B= Counter(B_prime).most_common(beam_size)
                B = {k[0]:k[1] for k in B}
            # use top one
            predSeq = Counter(B).most_common()[0][0].split('_')
            if predSeq == [pair[1] for pair in sentence]:
                correct+=1
            else:
                w=add(phi_1(sentence),w)
                w=subtract(dict(Counter(predSeq)),w)
                a=add(phi_1(sentence),a, coe=step/(epochs*len(train)))
                a=subtract(dict(Counter(predSeq)),a,coe=step/(epochs*len(train)))
            step-=1
        print('incorrect ', len(train)-correct)
    return a

# same as training
def predict_beam(test, w):
    predictions=[]
    for sentence in test:
        startList= [(sentence[0][0], tagList[i]) for i in range(len(tagList))]
        B={ i:(w[i] if i in w else 0) for i in startList}
        B=Counter(B).most_common(beam_size)
        B = {k[0][1]:k[1] for k in B}
        count = 0
        for pair in sentence[1:]:
            count += 1
            B_prime={}
            for b in B:
                for y in tagList:
                    wordSeq = [sentence[i][0] for i in range(count+1)]
                    tagSeq = b.split('_')+[y]
                    score = multiply_dict(Counter([(wordSeq[i], tagSeq[i]) for i in range(count+1)]),w)
                    B_prime['_'.join(tagSeq)] = score
            B= Counter(B_prime).most_common(beam_size)
            B = {k[0]:k[1] for k in B}
        predSeq = Counter(B).most_common()[0][0].split('_')
        predictions.append(predSeq)
    return predictions

# print timing & f1
if viterbi:
	mode = 'viterbi'
elif beam:
	mode = 'beam'

start = time.time()
if mode == 'viterbi':
	train_viterbi=train_viterbi(train,epoches=6)
	print('Training finished. Viterbi takes ',round(time.time()-start,4)/6,'sec for each epoch')
	evaluation(test, train_viterbi)
elif mode == 'beam':
    # specify beam size as global variable
	beam_size = 2
	start = time.time()
	w_beam = train_beam(train, 6)
	evaluation(test, w_beam)
	print('Training finished. Beam search takes ',round(time.time()-start,4)/6,'sec for each epoch')
