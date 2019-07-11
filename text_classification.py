from collections import Counter
import re
import os
import time
import random
from random import shuffle
import numpy as np
import argparse
import matplotlib.pyplot as plt

start = time.time()
parser = argparse.ArgumentParser()

parser.add_argument("infile")
args = parser.parse_args()
file_dir_sub = args.infile + '/txt_sentoken/'


# build model for each word
def n_grams(n,file_id):
    file_path = file_dir_sub + file_id
    with open(file_path) as f:
        text = f.read()
    if n=='unigram':
        bag_of_words = dict(Counter(re.sub("[^\w']"," ",text).split()))
        return bag_of_words
    elif n=='bigram':
        # use regex to match all bigrams which are in the form of "space + word + space + word"
        # in order to match the first bigram in an article,a space is added at the beginning of text
        bag_of_words = dict(Counter(re.sub("[^\w']"," ",text).split()))
        bag_of_words_bi = dict(Counter(re.findall(r'(?=( [a-zA-Z]+ [a-zA-Z]+))',' '+re.sub("[^\w']"," ",text))))
        bag_of_words.update(bag_of_words_bi)
        return bag_of_words
    elif n=='trigram':
        bag_of_words = dict(Counter(re.sub("[^\w']"," ",text).split()))
        #similar to bigram
        bag_of_words_tri = dict(Counter(re.findall(r'(?=( [a-zA-Z]+ [a-zA-Z]+ [a-zA-Z]+))',' '+re.sub("[^\w']"," ",text))))
        bag_of_words.update(bag_of_words_tri)
        return bag_of_words
        


# split text files to training set & testing set w.r.t filenames
def set_split(train_size, dir, tag):
    return [tag + i for i in os.listdir(dir + tag) if int(i[2:5])<train_size],[tag+i for i in os.listdir(dir+tag) if int(i[2:5])>=train_size]

def predict(weight, doc):
    return sign(sum(weight[word] * doc[word] for word in doc))


def sign(p):
    if p >=0 :
        return 1
    else:
        return -1


# dictionary a is introduced to keep track of every change of weight
# to avoid averaging thousands of weight dictionaries
# to find out more, visit https://svn.spraakdata.gu.se/repos/richard/pub/ml2014_web/m7.pdf
def train(init_weight, train_df, epoches, train_data, train_label):
    init_weight['█'] = 0
    w = init_weight.copy()
    a = init_weight.copy()
    step = epoches * len(train_df)
    accuList = []
    for epoch in range(1,epoches+1):
        random.seed(788)
        shuffle(train_df)
        for record in train_df:
            # training set is a list of tuples whose first element
            # is inverted index while the second one is the label
            doc=record[0]
            # add a black square in each index to stand for bias
            doc['█']=1
            label=record[1]
            p=predict(w,doc)
            if p != label:
                for k in doc:
                    w[k] += doc[k]*label
                    a[k] += doc[k]*label*step/(epoches*len(train_df))
            step -= 1
        accuList.append(evaluate4train(train_data, train_label, a))
    #return average weight a and a list of accuracies for each epoch accuList
    return a, accuList

# keep track of accuracy during learning progress
def evaluate4train(X_test, y_test, weight):
    test_result = np.array([predict(weight, d) for d in X_test])
    return sum(y_test == test_result)/len(y_test)

# evaluate the performance with 3 metrics: precision recall and F-score
def evaluate4test(X_test, y_test, weight):
    test_result = np.array([predict(weight, d) for d in X_test])
    accuracy = sum(y_test == test_result)/len(y_test)
    TP = sum((test_result==y_test)&(test_result==1))
    FP = sum((test_result!=y_test)&(test_result==1))
    FN = sum((test_result!=y_test)&(y_test==1))
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1: ', (2*precision*recall)/(precision+recall))

pos_train_path, pos_test_path = set_split (800, file_dir_sub, 'pos/')
neg_train_path, neg_test_path = set_split (800, file_dir_sub, 'neg/')


def pipeline(level, epoches):
    # select feature w.r.t the passed parameter
    print(level+' feature extraction ......')
    func = lambda x: n_grams(level,x)
    # apply function to the list of file path, list of dictionaries will be returned
    pos_train_set = list(map(func, pos_train_path))
    pos_test_set = list(map(func, pos_test_path))
    neg_train_set = list(map(func, neg_train_path))
    neg_test_set = list(map(func, neg_test_path))

    # initialize the weights for all grams to 0
    init_weight = set()
    for i in pos_test_set+neg_test_set+pos_train_set+neg_train_set:
        for j in i.keys():
            init_weight.add(j)
    init_weight = {i:0 for i in init_weight}

    # split train set & test set
    train_df = list(zip(*[pos_train_set+neg_train_set,[1]*800+[-1]*800]))
    train_data=pos_train_set+neg_train_set
    train_label=np.array([1]*800+[-1]*800)
    test_data=pos_test_set+neg_test_set
    test_label=np.array([1]*200+[-1]*200)
    print('train&test..........')
    # set number of epoches to 8
    weight, accuList=train(init_weight, train_df, epoches, train_data, train_label)
    evaluate4test(test_data, test_label, weight)
    #print most positive grams
    weight.pop('█')
    print(level + ' top 10 positive ========================')
    top10word=[]
    for k,v in Counter(weight).most_common(10):
        top10word.append(k)
    print(top10word)
    top10word = []
    print(level + ' top 10 negative ========================')
    weight = {k:-weight[k] for k in weight}
    for k,v in Counter(weight).most_common(10):
        top10word.append(k)
    print(top10word)
    #return a list of accuracies for plotting
    return [100 * i for i in accuList]

#set number of iterations
epoches=10

uniaccu = pipeline('unigram',epoches)
biaccu = pipeline('bigram',epoches)
triaccu = pipeline('trigram',epoches)
print(round(time.time()-start),'sec')
#plot a multiline chart
plt.plot(range(1,epoches+1), uniaccu,marker='o', markersize=5, color='skyblue', label='unigram')
plt.plot(range(1,epoches+1), biaccu,marker='o', markersize=5, color='magenta', label='bigram')
plt.plot(range(1,epoches+1), triaccu,marker='o', markersize=5, color='cyan', label='trigram')
plt.title('classification accuracy using different features & epoches')
plt.xlabel('epoches')
plt.ylabel('accuracy (%)')
plt.legend()
plt.show()






