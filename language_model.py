from collections import Counter
import re
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("corpus")
parser.add_argument("questions")
args = parser.parse_args()
text_path = args.corpus
questions_path=args.questions

#open the questions file
# questions_path = '/Users/Patrick/Downloads/questions.txt'
with open(questions_path) as f:
    questions = f.readlines()
content = [x.strip() for x in questions]

# split questions file to questions + candidates(options)
questions=[]
options=[]
for i in content:
    questions.append(i.split(':')[0])
    options.append(i.split(':')[1])
options=[(i.split('/')[0][1:], i.split('/')[1]) for i in options]

#specify the solution for questions
keys=['whether', 'through', 'piece', 'court', 'allowed', 'check', 'hear', 'cereal', 'chews', 'sell']

# use regex to match tokens in corpus
# text_path='/Users/Patrick/Downloads/news-corpus-500k.txt'
with open(text_path) as f:
    line = f.read()
# replace \n with <\s> <s> because each line is a sentence
line=line.replace('\n', '<\s> <s> ')
# add start symbol and end symbol at the begining and ending respectively
line=re.sub(' +',' ',' <s> '+re.sub("[^\w'<\\\>-]"," ",line)+'<\s>')
# delete all punctuations except - \ and '
uni_list= re.sub("[^\w'<\\\>-]", " ", line).split()
uni_sum = len(uni_list)
uni_dict = dict(Counter(uni_list))

#match bigrams
bigram_list=re.findall(r'(?=( [a-zA-Z<\'>-]+ [a-zA-Z<\\\'>-]+))', line)
bi_dict= dict(Counter(bigram_list))
bi_dict= {k[1:]: v for k, v in bi_dict.items()}


# feature: bigram or unigram
# lmd: smoothing coefficient lambda

def LM(feature, lmd):
    answers=[]
    accuracy=0
    for i in range(len(questions)):
        p=[]
        for option in options[i]:
            sentence = questions[i].replace('____', option)
            # tokenize the question
            sentence_uni = re.sub("[^\w'<\\\>-]", " ", ' <s> '+sentence+'<\s>').split()
            sentence_bi= [sentence_uni[i] + ' '+ sentence_uni[i+1] for i in range(len(sentence_uni)-1)]
            # initialize probability as 1 for each candidate in each question
            prob = 1.0
            if feature == 'Bigram':
                for word in sentence_bi:
                    if lmd == True:
                        # conditional probability count(bigram)+1/count(unigram)+V
                        if word in bi_dict:
                            prob *= (bi_dict[word]+lmd)/(uni_dict[word.split(' ')[0]]+lmd*len(uni_dict))
                        else:
                            prob *= lmd/(lmd*len(uni_dict))
                    else:
                        if word in bi_dict:
                            prob *= bi_dict[word]/uni_dict[word.split(' ')[0]]
                        else:
                            prob *=0
            elif feature == 'Unigram':
                for word in sentence_uni:
                    if word in uni_dict:
                        prob *= (uni_dict[word] / uni_sum)
                    else:
                        prob *= 0
            p.append(prob)
        # when return 0 probabilities for both candidates, the answer is incorrect
        if p[0] == p[1] and p[1] == 0:
            answers.append('both zero')
        elif p[0] == p[1]:
        # when tie with non-zero probs, half correct
            answers.append('tie')
            accuracy += 0.5
        # otherwise compare the answer with given solution
        elif p[0] > p[1]:
            answers.append(options[i][0])
            accuracy += (options[i][0] == keys[i])
        elif p[0] < p[1]:
            answers.append(options[i][1])
            accuracy += (options[i][1] == keys[i])
    print('accuracy = ', accuracy/len(keys))
    print('answers : ', answers)
print('Unigram:')
LM('Unigram', False)
print('Bigram: ')
LM('Bigram', False)
print('Bigram + Smoothing:')
LM('Bigram', 1)