# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 18:39:59 2018

@author: Titli Sarkar 
ULID# C00222141
CSCE 588 Neural Network Project 
Spring 2018

Probelm Statement: Classify the emotions in tweets.
Models Tried: LSTM, BiLSTM
"""
# import libraries here which are needed
import pandas as pd
import numpy as np
import os
from keras.utils import to_categorical
from gensim.scripts.glove2word2vec import glove2word2vec 
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
import html, re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
#nltk.download('punkt')
#nltk.download('stopwords')


path = "E:\\Studies\\Ph.D\\Sem 4\\CSCE 619\\SemEval2018Task\\" # path to the files: train, dev, test in csv format
GLOVE_DIR = "E:\\Studies\\Ph.D\\Sem 4\\CSCE 619\\SemEval2018Task\\glove.twitter.27B\\" # path to Glove file

input_dim = 100  # as glove embedding vector dim = 100

# helper function: put values of dict to a ndarray
def dict_to_array(dic):
    return [v for _, v in dic.items()]

# read file  
def ReadCSV(datafile, labelfile):
    inputdata = pd.io.parsers.read_csv(open(datafile, "r"),delimiter=",") 
    data = inputdata.as_matrix() # get data as matrix 
    label = np.loadtxt(open(labelfile, "rb"),delimiter=",") #get label as a list
    return data, label

# helper function: remove punctuations
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation) # use nltk 

# helper function: remove stopwords
def stopwordsremoval(sentence):
    stopwords_removed = [word for word in sentence.split(' ') if word not in stopwords.words('english')] # use nltk 
    return stopwords_removed

# helper function: process a string; only keep words
def clean_str(string):
    string = html.unescape(string)
    string = string.replace("\\n", " ")
    #string = string.replace("_NEG", "")
    #string = string.replace("_NEGFIRST", "")
    string = re.sub(r"@[A-Za-z0-9_s(),!?\'\`]+", "", string) #removes @---, 
    string = re.sub(r"\*", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ,", string)
    string = re.sub(r"!", " !", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ?", string)
    string = re.sub(r"\s{2,}", " ", string)
    return stopwordsremoval(strip_punctuation(string.strip().lower()))

#Step 1: Read data and process data
def preprocessing(data): ## we will return everything as dictionaries; key=id, value = tweets/labels/intensity values
    corpus = [] # data
    train_sentences=data
    
    # adding processed tweets in a dict
    for item in train_sentences:
        sentence = sent_tokenize(item) # sentence tokenize, list of sentences
        processed_tweet = []
        for sen in sentence:
            sen1=""
            sen1 = clean_str(sen)
            processed_tweet = processed_tweet+sen1
        corpus.append(processed_tweet)
    return corpus

# helper function: converts input to one-hot encoded vector
def one_hot_encoding(y):
    y = to_categorical(y) # Converts a class vector (integers) to binary class matrix
    return y[:,1:] #remove extra zero column at the first


# read data file (...............code starts here..................)
# Step 1 -->
print(os.getcwd())
list1 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-test\\2018-EI-reg-En-anger-test.txt',sep='\t')['Tweet'].tolist()
list2 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-test\\2018-EI-reg-En-fear-test.txt',sep='\t')['Tweet'].tolist()
list3 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-test\\2018-EI-reg-En-joy-test.txt',sep='\t')['Tweet'].tolist()
list4 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-test\\2018-EI-reg-En-sadness-test.txt',sep='\t')['Tweet'].tolist()
list5 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-train\\2018-EI-reg-En-anger-train.txt',sep='\t')['Tweet'].tolist()
list6 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-train\\2018-EI-reg-En-fear-train.txt',sep='\t')['Tweet'].tolist()
list7 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-train\\2018-EI-reg-En-joy-train.txt',sep='\t')['Tweet'].tolist()
list8 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-train\\2018-EI-reg-En-sadness-train.txt',sep='\t')['Tweet'].tolist()
list9 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-dev\\2018-EI-reg-En-anger-dev.txt',sep='\t')['Tweet'].tolist()
list10 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-dev\\2018-EI-reg-En-fear-dev.txt',sep='\t')['Tweet'].tolist()
list11 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-dev\\2018-EI-reg-En-joy-dev.txt',sep='\t')['Tweet'].tolist()
list12 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-dev\\2018-EI-reg-En-sadness-dev.txt',sep='\t')['Tweet'].tolist()

list11 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-test\\2018-EI-reg-En-anger-test.txt',sep='\t')['Affect Dimension'].tolist()
list22 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-test\\2018-EI-reg-En-fear-test.txt',sep='\t')['Affect Dimension'].tolist()
list33 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-test\\2018-EI-reg-En-joy-test.txt',sep='\t')['Affect Dimension'].tolist()
list44 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-test\\2018-EI-reg-En-sadness-test.txt',sep='\t')['Affect Dimension'].tolist()
list55 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-train\\2018-EI-reg-En-anger-train.txt',sep='\t')['Affect Dimension'].tolist()
list66 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-train\\2018-EI-reg-En-fear-train.txt',sep='\t')['Affect Dimension'].tolist()
list77 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-train\\2018-EI-reg-En-joy-train.txt',sep='\t')['Affect Dimension'].tolist()
list88 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-train\\2018-EI-reg-En-sadness-train.txt',sep='\t')['Affect Dimension'].tolist()
list99 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-dev\\2018-EI-reg-En-anger-dev.txt',sep='\t')['Affect Dimension'].tolist()
list1010 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-dev\\2018-EI-reg-En-fear-dev.txt',sep='\t')['Affect Dimension'].tolist()
list1111 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-dev\\2018-EI-reg-En-joy-dev.txt',sep='\t')['Affect Dimension'].tolist()
list1212 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-dev\\2018-EI-reg-En-sadness-dev.txt',sep='\t')['Affect Dimension'].tolist()

input_data = list1+list2+list3+list4+list5+list6+list7+list8+list9+list10+list11+list12
print(list11)
print("input_data Shape:", len(input_data))

train_data = preprocessing(input_data)
#train_data = dict_to_array(preprocessed_data)

# word2vec model generate
def word2vec_model(sentences):
    model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4, sg=1) # train model , size(10-100)
    words = list(model.wv.vocab) # summarize vocabulary
    print(len(words))
    model.save('w2v_model.bin') # save model

# glove model generate
def glove_model():

    glove_input_file = 'glove.twitter.27B.100d.txt'
    word2vec_output_file = 'word2vec.twitter.27B.100d.txt'
    
    if not os.path.isfile(word2vec_output_file):
        glove2word2vec(glove_input_file, word2vec_output_file)
        glove2vec = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        glove2vec.save('glove_model.bin') # save model

# func call
word2vec_model(train_data) 
glove_model()
w2v_model = Word2Vec.load('w2v_model.bin') # load model
glove_model = KeyedVectors.load_word2vec_format('w2v_model.bin') # load model

#print(w2v_model) # summarize the loaded model
print(len(w2v_model.wv['angry'])) # access vector for one word
print(w2v_model.wv.most_similar('angry'))

# fit a 2d PCA model to the vectors (to visualize distrubution of words in w2v_model)
'''X = w2v_model[w2v_model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
plt.scatter(result[:, 0], result[:, 1])
words = list(w2v_model.wv.vocab)
print(len(words))
for i, word in enumerate(words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()
'''
