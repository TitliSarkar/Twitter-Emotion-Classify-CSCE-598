# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:39:59 2018
@author: Titli Sarkar 
ULID# C00222141
CSCE 598 Deep Learning Project 
Spring 2018

Probelm Statement: Classify the emotions in tweets.
Models Tried: LSTM, BiLSTM, CNN-LSTM, GRU
"""
# import libraries here which are needed
import pandas as pd
import numpy as np
import os

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, GRU, Bidirectional, Convolution1D, MaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam
#from gensim.scripts.glove2word2vec import glove2word2vec 
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from sklearn.metrics import confusion_matrix
import html, re
#import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
import matplotlib.pyplot as plt
import gensim.models as gsm
#import emoji
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from keras.utils.vis_utils import plot_model
#import pydot, graphviz

#nltk.download('punkt')
#nltk.download('stopwords')
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\'

path = "E:\\Studies\\Ph.D\\Sem 4\\CSCE 619\\SemEval2018Task\\" # path to the files: train, dev, test in csv format
GLOVE_DIR = "E:\\Studies\\Ph.D\\Sem 4\\CSCE 619\\SemEval2018Task\\glove.twitter.27B\\" # path to Glove file

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
    corpus = [] # data , a list of lists
    train_sentences=data
    
    # adding processed tweets in a dict
    for item in train_sentences:
        sentence = sent_tokenize(item) # sentence tokenize, list of sentences
        processed_tweet = [] # a full tweet (may be with multiple sentences) in processed form
        for sen in sentence:
            sen1=""
            sen1 = clean_str(sen)
            processed_tweet = processed_tweet+sen1
        corpus.append(processed_tweet)
    return corpus

# helper function: converts a list of 'anger', 'fear', 'joy', 'sadness' to 1-4 labels
def enumerate_list(input_list):
    e_dict = {}
    idx = 0
    for i in list(set(input_list)):
        #print (i)
        e_dict[i] = idx+1
        idx += 1
    #print(e_dict)
    e_list = [e_dict[x] for x in input_list]
    return(e_list)
    
# helper function: converts input to one-hot encoded vector
def one_hot_encoding(y):
    y = to_categorical(y) # Converts a class vector (integers) to binary class matrix
    print(type(y))
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

list111 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-test\\2018-EI-reg-En-anger-test.txt',sep='\t')['Affect Dimension'].tolist()
list222 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-test\\2018-EI-reg-En-fear-test.txt',sep='\t')['Affect Dimension'].tolist()
list333= pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-test\\2018-EI-reg-En-joy-test.txt',sep='\t')['Affect Dimension'].tolist()
list444 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-test\\2018-EI-reg-En-sadness-test.txt',sep='\t')['Affect Dimension'].tolist()

list555 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-train\\2018-EI-reg-En-anger-train.txt',sep='\t')['Affect Dimension'].tolist()
list666 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-train\\2018-EI-reg-En-fear-train.txt',sep='\t')['Affect Dimension'].tolist()
list777 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-train\\2018-EI-reg-En-joy-train.txt',sep='\t')['Affect Dimension'].tolist()
list888 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-train\\2018-EI-reg-En-sadness-train.txt',sep='\t')['Affect Dimension'].tolist()

list999 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-dev\\2018-EI-reg-En-anger-dev.txt',sep='\t')['Affect Dimension'].tolist()
list101010 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-dev\\2018-EI-reg-En-fear-dev.txt',sep='\t')['Affect Dimension'].tolist()
list111111 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-dev\\2018-EI-reg-En-joy-dev.txt',sep='\t')['Affect Dimension'].tolist()
list121212 = pd.read_csv(os.getcwd()+'\\data\\2018-EI-reg-En-dev\\2018-EI-reg-En-sadness-dev.txt',sep='\t')['Affect Dimension'].tolist()

train_data, train_label = preprocessing(list5+list6+list7+list8), enumerate_list(list555+list666+list777+list888) #list5+list6+list7+list8 , list55+list66+list77+list88 
dev_data, dev_label = preprocessing(list9+list10+list11+list12), enumerate_list(list999+list101010+list111111+list121212) #list9+list10+list11+list12 , list99+list1010+list1111+list1212 
#test_data, test_label = preprocessing(list1+list2+list3+list4), enumerate_list(list111+list222+list333+list444) #list1+list2+list3+list4 , list11+list22+list33+list44 
#note: full test dataset is too huge-> memory error, so use part of it
test_data_list = pd.read_csv(os.getcwd()+'\\data\\EI-reg-En-part-test.csv')['Tweet'].tolist()
test_label_list = pd.read_csv(os.getcwd()+'\\data\\EI-reg-En-part-test.csv')['Affect Dimension'].tolist()
test_data, test_label = preprocessing(test_data_list),test_label_list

print("Train shape:", len(train_data), len(train_label))
print("Validation shape:", len(dev_data), len(dev_label))
print("Test shape:", len(test_data), len(test_label))

# Loading all models
glove_model = KeyedVectors.load_word2vec_format('word2vec.twitter.27B.100d.txt', binary=False) # load Glove model
w2v_model = Word2Vec.load('w2v_model.bin') # load word2vec model
e2v_model = gsm.KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True) # load emoji2vec model 
print("All Models Loaded!")

# word embedding data with glove pretrained model and real word2vec/w2v
input_data = np.concatenate((train_data, dev_data, test_data))
max_sequence_length = max([len(x) for x in input_data]) # find the length of longest twitter
print("Max twitter length:", max_sequence_length)
print("input_data shape:", len(input_data))

# Find embedding for corpus
def embedding(data, max_len):
    data_eb = [] # saves embedding for full corpus
    for i in range(len(data)):
        row_eb = [] # saves embedding for each row/tweet
        for j, token in enumerate(data[i]):
            token_eb =[] # each word embedding should be dim(glove+w2v+e2v)
            if (token in glove_model):{token_eb.append(glove_model[token])}
            else:{token_eb.append(np.zeros(100))}
                
            if (token in w2v_model):{token_eb.append(w2v_model[token])}
            else:{token_eb.append(np.zeros(100))}
                
            if (token in e2v_model):{token_eb.append(e2v_model[token])}
            else:{token_eb.append(np.zeros(300))}
            
            token_eb = [y for x in token_eb for y in x] #flatten
            for n in range(len(token_eb), 500): # append 0 to each embedding equidim
                token_eb.append(0)
            token_eb = np.array(token_eb) #numpy.ndarray (500,)
            row_eb.append(token_eb) #list
        data_eb.append(row_eb)
    data_eb = pad_sequences(data_eb, maxlen=max_len) #zero padding for making corpus equidimensional
    print(type(data_eb), data_eb.shape) #numpy.ndarray
    return data_eb

# Step 3: Find word embeddings of data 
train_data = embedding(train_data, max_sequence_length)
dev_data = embedding(dev_data, max_sequence_length)
test_data = embedding(test_data, max_sequence_length)

print("Train embedding shape:", train_data.shape, len(train_label))
print("Dev embedding shape:", dev_data.shape, len(dev_label))
print("Test embedding shape:", test_data.shape, len(test_label))

# convert label to one-hot vector
labels = np.concatenate((train_label, dev_label, test_label))
number_classes = len(np.unique(labels))
print("Number of output classes:", number_classes)
y_oh = one_hot_encoding(labels)

train_label = one_hot_encoding(np.asarray(train_label))
dev_label = one_hot_encoding(np.asarray(dev_label))
test_label = one_hot_encoding(np.asarray(test_label))
print("One-hot encoded labels shape (train, validation, test):", train_label.shape, dev_label.shape, test_label.shape)

input_dim = np.max([train_data.shape[2], dev_data.shape[2], test_data.shape[2]])  #100  # as glove embedding vector dim = 100
print(input_dim)

# Finalize data to be passed: Concat to train on both train+dev set, only validate on test set
X_train = np.concatenate((train_data, dev_data)) # train data
y_train = np.concatenate((train_label, dev_label)) # train label
print("Training size:", X_train.shape)
print("Test size:", test_data.shape)

# Step 4: Create neural network models and use them

# *** implementing model CNN-LSTM ***
    # defining model
def compile_model_cnn_lstm(input_dim, latent_dim, num_class):
    '''Create CNN-LSTM model
    Args:
        input_dim (int): dim of embedding vector
        latent_dim (int): dim of output from LSTM layer
        num_class (int): number output class
    '''
    inputs = Input(shape=(None, input_dim)) # create input
    
    conv = Convolution1D(1024, kernel_size=1, padding='valid', activation='tanh')(inputs) 
    conv = MaxPooling1D(pool_size=2)(conv)
    print(conv.shape) # (? ,? ,1024)
    
    lstm = LSTM(latent_dim)(conv) # create CNN-LSTM layer with #units = latent_dim
    print(lstm.shape) #(?,64)
    drop = Dropout(0.3)(lstm) # define 30% dropout
    out = Dense(num_class, activation='softmax')(drop)  #define output layer with output dimension=mun_class
    
    model = Model(inputs, out) # create model; this is a logistic regression in Keras
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) # compile model with defined parameters
    model.summary()
    plot_model(model, to_file='CNN-LSTM_model_plot.png', show_shapes=True, show_layer_names=True)
    return model
  # running model
def run_cnn_lstm(epochs, batch_size=128):
    # create cnn-lstm model
    model = compile_model_cnn_lstm(input_dim, 64, number_classes)

    checkpointer = ModelCheckpoint(filepath='twitter-emotion-cnn-lstm.h5', verbose=1, save_best_only=True)
    model.fit(X_train, y_train, validation_data=(test_data, test_label), callbacks=[checkpointer], 
              shuffle=True, epochs=epochs, batch_size=batch_size, verbose=1)
    
    y_pred = model.predict(test_data, batch_size=batch_size)
    
    y_actual = np.argmax(test_label,axis=1) # actual test labels, get back from one_hot vectors
    y_predicted = np.argmax(y_pred, axis=1) # predicted test labels, get back from one_hot vectors
    
    confusionMatrix=confusion_matrix(y_actual,y_predicted)     # Show confusion matrix (actual vs. predicted)
    print ("\nConfusion Matrix on test data [(4x4) for 4 output labels of 'anger', 'fear', 'joy', 'sadness']: ")
    print(confusionMatrix)
    variation = np.absolute(y_predicted - y_actual)     # get the diference of predicted and actual class_labels
    print ("CNN-LSTM accuracy=", np.mean(variation))
    print("CNN-LSTM perason=",np.absolute(np.corrcoef(y_predicted,y_actual)[0, 1]))
    print("CNN-LSTM F1-score=", f1_score(y_actual, y_predicted, average="macro"))
    print("CNN-LSTM precision-score=", precision_score(y_actual, y_predicted, average="macro"))
    print("CNN-LSTM recall-score=", recall_score(y_actual, y_predicted, average="macro"))    

# *** implementing model biLSTM ***
    # defining model
def compile_model_bi_lstm(input_dim, latent_dim, num_class):
    '''Create BiLSTM model
    Args:
        input_dim (int): dim of embedding vector
        latent_dim (int): dim of output from LSTM layer
        num_class (int): number output class
    '''
    inputs = Input(shape=(None, input_dim)) # create input
    bilstm = Bidirectional(LSTM(latent_dim))(inputs) # create BiLSTM layer with #units = latent_dim
    drop = Dropout(0.3)(bilstm) # define 30% dropout
    out = Dense(num_class, activation='softmax')(drop) # define output layer with output dimension=mun_class
    model = Model(inputs, out) # create model; this is a logistic regression in Keras
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) # compile model with defined parameters
    model.summary() # print model summary
    plot_model(model, to_file='biLSTM_model_plot.png', show_shapes=True, show_layer_names=True)
    return model
# running model
def run_bi_lstm(epochs, batch_size=128):
    print ("\n\nRunning bi-LSTM model.......")
    model = compile_model_bi_lstm(input_dim, 64, number_classes) #Build Model

    checkpointer = ModelCheckpoint(filepath='twitter-emotion-bi_lstm.h5', verbose=1, save_best_only=True) # Save the model after every epoch; precaution in case of system failure

    model.fit(X_train, y_train, validation_data=(test_data, test_label), callbacks=[checkpointer], 
              shuffle=True, epochs=epochs, batch_size=batch_size, verbose=1) # train model with train data and validate on test data
    
    y_pred = model.predict(test_data, batch_size=batch_size) # get the prediction result on test data

    y_actual = np.argmax(test_label,axis=1) # actual test labels, get back from one_hot vectors
    y_predicted = np.argmax(y_pred, axis=1) # predicted test labels, get back from one_hot vectors
    
    confusionMatrix=confusion_matrix(y_actual,y_predicted)     # Show confusion matrix (actual vs. predicted)
    print ("\nConfusion Matrix on test data [(4x4) for 4 output labels of 'anger', 'fear', 'joy', 'sadness']: ")
    print(confusionMatrix)
    variation = np.absolute(y_predicted - y_actual)     # get the diference of predicted and actual class_labels
    print ("biLSTM accuracy=", np.mean(variation))
    print("biLSTM perason=",np.absolute(np.corrcoef(y_predicted,y_actual)[0, 1]))
    print("biLSTM F1-score=", f1_score(y_actual, y_predicted, average="macro"))
    print("biLSTM precision-score=", precision_score(y_actual, y_predicted, average="macro"))
    print("biLSTM recall-score=", recall_score(y_actual, y_predicted, average="macro"))

# *** implementing model LSTM ***
    # defining model
def compile_model_lstm(input_dim, latent_dim, num_class):
    '''Create LSTM model
    Args:
        input_dim (int): dim of embedding vector
        latent_dim (int): dim of output from LSTM layer
        num_class (int): number output class
    '''
    inputs = Input(shape=(None, input_dim)) # create input
    lstm = LSTM(latent_dim)(inputs) # create LSTM layer with #units = latent_dim
    drop = Dropout(0.5)(lstm) # define dropout
    # Dense1
    #z = Dense(1024, activation='relu')(drop)
    #z = Dropout(0.3)(z)
    # Dense2
    #z = Dense(256, activation='relu')(z)
    #z = Dropout(0.3)(z)
    
    out = Dense(num_class, activation='softmax')(drop) # z or drop # define output layer with output dimension=mun_class
    model = Model(inputs, out) # create model; this is a logistic regression in Keras
    rmsprop = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy']) # compile model with defined parameters
    model.summary() # print model summary
    plot_model(model, to_file='LSTM_model_plot.png', show_shapes=True, show_layer_names=True)
    return model
  # running model
def run_lstm(epochs, batch_size=128):
    print ("\n\nRunning LSTM model.......")
    model = compile_model_lstm(input_dim, 64, number_classes) # create lstm model

    checkpointer = ModelCheckpoint(filepath='twitter-emotion-lstm.h5', verbose=1, save_best_only=True) # Save the model after every epoch; precaution in case of system failure
    
    model.fit(X_train, y_train, validation_data=(test_data, test_label), callbacks=[checkpointer], 
              shuffle=True, epochs=epochs, batch_size=batch_size, verbose=1)     # train model with train data and validate on test data

    y_pred = model.predict(test_data, batch_size=batch_size)    # get the prediction result on test data
    
    y_actual = np.argmax(test_label,axis=1) # actual test labels, get back from one_hot vectors
    y_predicted = np.argmax(y_pred, axis=1) # predicted test labels, get back from one_hot vectors
    
    confusionMatrix=confusion_matrix(y_actual,y_predicted)     # Show confusion matrix (actual vs. predicted)
    print ("\nConfusion Matrix on test data [(4x4) for 4 output labels of 'anger', 'fear', 'joy', 'sadness']: ")
    print(confusionMatrix)
    variation = np.absolute(y_predicted - y_actual)     # get the diference of predicted and actual class_labels
    print("LSTM accuracy=", np.mean(variation))
    print("LSTM perason=",np.absolute(np.corrcoef(y_predicted,y_actual)[0, 1]))
    print("LSTM F1-score=", f1_score(y_actual, y_predicted, average="macro"))
    print("LSTM precision-score=", precision_score(y_actual, y_predicted, average="macro"))
    print("LSTM recall-score=", recall_score(y_actual, y_predicted, average="macro"))
    
    '''plt.figure(1)
    plt.plot(variation[:400])
    plt.xlabel('LSTM 0-400')
    plt.show()
    
    plt.figure(2)
    plt.plot(variation[400:800])
    plt.xlabel('LSTM 400-800')
    plt.show()
    
    plt.figure(3)
    plt.plot(variation[800:1200])
    plt.xlabel('LSTM 800-1200')
    plt.show()
    
    plt.figure(4)
    plt.plot(variation[1200:1600])
    plt.xlabel('LSTM 1200-1600')
    plt.show()
    
    plt.figure(5)
    plt.plot(variation[1600:2000])
    plt.xlabel('LSTM 1600-2000')
    plt.show()'''

# *** implementing model RNN-LSTM ***
    # defining model
def compile_model_gru(input_dim, latent_dim, num_class):
    inputs = Input(shape=(None, input_dim))
    
    #rnn = SimpleRNN(256, activation='tanh')(inputs)
    gru = GRU(latent_dim)(inputs)
    drop = Dropout(0.3)(gru)
    
    # Dense1
    z = Dense(256, activation='tanh')(drop)
    z = Dropout(0.3)(z)
    
    out = Dense(num_class, activation='softmax')(z)
    model = Model(inputs, out)
    rmsprop = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])  #mean_absolute_error, rmsprop
    model.summary()
    plot_model(model, to_file='GRU_model_plot.png', show_shapes=True, show_layer_names=True)
    return model
    # running model
def run_gru(epochs, batch_size=128):
    # create cnn-lstm model
    model = compile_model_gru(input_dim, 64, number_classes)

    checkpointer = ModelCheckpoint(filepath='twitter-emotion-gru.h5', verbose=1, save_best_only=True)
    model.fit(X_train, y_train, validation_data=(test_data, test_label), callbacks=[checkpointer], 
              shuffle=True, epochs=epochs, batch_size=batch_size, verbose=1)
    
    y_pred = model.predict(test_data, batch_size=batch_size)
    
    y_actual = np.argmax(test_label,axis=1) # actual test labels, get back from one_hot vectors
    y_predicted = np.argmax(y_pred, axis=1) # predicted test labels, get back from one_hot vectors
    
    confusionMatrix=confusion_matrix(y_actual,y_predicted)     # Show confusion matrix (actual vs. predicted)
    print ("\nConfusion Matrix on test data [(4x4) for 4 output labels of 'anger', 'fear', 'joy', 'sadness']: ")
    print(confusionMatrix)
    variation = np.absolute(y_predicted - y_actual)     # get the diference of predicted and actual class_labels
    print ("GRU accuracy=", np.mean(variation))
    print("GRU perason=",np.absolute(np.corrcoef(y_predicted,y_actual)[0, 1]))
    print("GRU F1-score=", f1_score(y_actual, y_predicted, average="macro"))
    print("GRU precision-score=", precision_score(y_actual, y_predicted, average="macro"))
    print("GRU recall-score=", recall_score(y_actual, y_predicted, average="macro"))    

# func calls
#for i in range(2):
run_lstm(10)
run_bi_lstm(10)
run_cnn_lstm(10)
run_gru(10)
#print(i)
