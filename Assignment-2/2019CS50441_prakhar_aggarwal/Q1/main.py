import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import re
from operator import itemgetter

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.util import skipgrams
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def _stem(doc, p_stemmer, en_stop, return_tokens):
    tokens = word_tokenize(doc.lower())
    stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    stemmed_tokens = map(lambda token: p_stemmer.stem(token), stopped_tokens)
    if not return_tokens:
        return ' '.join(stemmed_tokens)
    return list(stemmed_tokens)

def getStemmedDocuments(docs, return_tokens=False):
    """
        Args:
            docs: str/list(str): document or list of documents that need to be processed
            return_tokens: bool: return a re-joined string or tokens
        Returns:
            str/list(str): processed document or list of processed documents
        Example: 
            new_text = "It is important to by very pythonly while you are pythoning with python. \
                All pythoners have pythoned poorly at least once."
            print(getStemmedDocuments(new_text))
        Reference: https://pythonprogramming.net/stemming-nltk-tutorial/
    """
    en_stop = set(stopwords.words('english'))
    ps = PorterStemmer()
    if isinstance(docs, list):
        output_docs = []
        for item in docs:
            output_docs.append(_stem(item, ps, en_stop, return_tokens))
        return output_docs
    else:
        return _stem(docs, ps, en_stop, return_tokens)

# nltk.download()

LaplaceSmoothing = 1
BASE_DIR = "../"
# train_path = os.path.join(BASE_DIR, 'data', 'reviews_Digital_Music_5.json', 'Music_Review_train.json')
# test_path  = os.path.join(BASE_DIR, 'data', 'reviews_Digital_Music_5.json', 'Music_Review_test.json')

train_path = sys.argv[1]
test_path  = sys.argv[2]


def load_data(filename):
    return pd.read_json(filename, lines=True)

def clean_data(line):
    line = line.strip().lower()
    line = re.sub(r'[^\w\s]','',line)
    line = re.sub('\r?\n',' ',line)
    return line

def preprocessing(df, stemming):
    ColumnsToDrop = ['reviewerID', 'asin', 'reviewerName', 'unixReviewTime', 'reviewTime']
    df = df.drop(ColumnsToDrop, axis=1)
    
    df['reviewText'] = df['reviewText'].apply(lambda x: clean_data(x))
    df['summary']    = df['summary'].apply(lambda x: clean_data(x))
    
    if stemming:
        df['reviewText'] = df['reviewText'].apply(lambda x: getStemmedDocuments(x))
        df['summary']    = df['summary'].apply(lambda x: getStemmedDocuments(x))
    return df

def Vocab_generation(df1, df2):
    VocabSize = 0
    Vocab = dict()
    
    min_val = min(df2)
    max_val = max(df2)
    
    VocabClassSize = [0]*(max_val - min_val + 1)
    ExampleSize    = [0]*(max_val - min_val + 1)    
        
    for i,review in enumerate(df1):
        for word in review.split():
            if len(word) > 0 and word not in Vocab:
                Vocab[word] = 1
                VocabSize  += 1

    VocabClass = []
    for i in range(max_val - min_val + 1):
        d = Vocab.copy()
        VocabClass.append(d)
    
    for i,review in enumerate(df1):
        ExampleSize[df2[i]-min_val] += 1
        for word in review.split():
            if len(word) > 0:
                VocabClassSize[df2[i]-min_val] += 1
                VocabClass[df2[i]-min_val][word] += 1
    
    return Vocab, VocabClass, VocabSize, VocabClassSize, ExampleSize

def predict(X,Y,phi,VocabClass,VocabClassSize):
    Y_pred = [0]*(Y.shape[0])
    count = 0
    minval = min(Y)
    
    for i,review in enumerate(X):
        class_prob = [0.0]*len(VocabClass)

        for word in review.split():
            for j in range(len(VocabClass)):
                if word in VocabClass[j]:
                    class_prob[j] += VocabClass[j][word]
                else:
                    class_prob[j] += (float(LaplaceSmoothing)/float(VocabSize))
                
        for j in range(len(VocabClass)):
            class_prob[j] += phi[j]
            
        class_label, element = max(enumerate(class_prob), key=itemgetter(1))
        class_label += minval
        
        Y_pred[i]   = class_label
        
        if class_label == Y[i]:
            count+=1
        
    return count, Y_pred

def f1_score_metric(A,B):
    conf_mat = np.zeros((5,5));
    for i in range(B.shape[0]):
        conf_mat[int(B[i]-1)][int(A[i]-1)] += 1;
    
    precision = np.zeros(5);
    recall = np.zeros(5);
    f1 = np.zeros(5);

    for i in range(5):
        precision[i] = conf_mat[i,i]/(np.sum(conf_mat, axis = 0)[i])
        recall[i] = conf_mat[i,i]/(np.sum(conf_mat, axis = 1)[i])
        f1[i] = (2*precision[i]*recall[i])/(precision[i]+recall[i]);

    return f1

# Random Guessing
def random_guessing(Y_test):
    Y_pred = np.zeros(len(Y_test))
    count = 0
    for i in range (Y_test.shape[0]):
        class_label = np.random.randint(1,6)
        Y_pred[i]   = class_label
        if(class_label == Y_test[i]):
            count+=1
    return count, Y_pred

# Majority Prediction
def majority_prediction(Y_test,phi):
    Y_pred = np.zeros(len(Y_test))
    count = 0
    class_label, element = max(enumerate(phi), key=itemgetter(1))
    class_label += 1
    
    for i in range (Y_test.shape[0]):        
        Y_pred[i]   = class_label
        if(class_label == Y_test[i]):
            count+=1
    return count, Y_pred

def bigrams(X):
    for i in range(X.shape[0]):
        review = X[i]
        token  = nltk.word_tokenize(review)
        bigrams = list(ngrams(token,2))
        bigrams = list(map(lambda x: "_".join(x), bigrams))
        X[i] = " ".join(bigrams)
    return X

def gen_skipgrams(X,skip_dist=1):
    for i in range(X.shape[0]):
        review = X[i].split()
        sg = list(skipgrams(review,2,skip_dist))
        sg = list(map(lambda x: "_".join(x), sg))
        X[i] = " ".join(sg)
    return X

PART = sys.argv[3]

if PART == 'a':
    print("Part A")
    train_data = preprocessing(load_data(train_path), False)
    test_data  = preprocessing(load_data(test_path), False) # False implies not to perform stemming

    X_train = train_data['reviewText'].copy()
    Y_train = train_data['overall'].copy()

    X_test = test_data['reviewText'].copy()
    Y_test = test_data['overall'].copy()
    
    print("Vocabulary Generation")
    
    Vocab, VocabClass, VocabSize, VocabClassSize, phi = Vocab_generation(X_train, Y_train)
    
    for i in range(len(VocabClass)):
        for x in VocabClass[i]:
            VocabClass[i][x] = np.log(float(VocabClass[i][x])/float(VocabClassSize[i] + VocabSize))
    
    for i in range(len(VocabClass)):
        phi[i] = np.log(float(phi[i])/float(len(X_train)))
        
    correct, Y_pred_train = predict(X_train, Y_train, phi, VocabClass, VocabClassSize)
    print("Train Set Accuracy = {}%".format(round(100*float(correct)/float(len(X_train)),2)))

    correct, Y_pred_test = predict(X_test, Y_test, phi, VocabClass, VocabClassSize)
    print("Test Set Accuracy  = {}%".format(round(100*float(correct)/float(len(X_test)),2)))

elif PART == 'b':
    print('Part B')
    train_data = preprocessing(load_data(train_path), False)
    test_data  = preprocessing(load_data(test_path), False) # False implies not to perform stemming

    X_train = train_data['reviewText'].copy()
    Y_train = train_data['overall'].copy()

    X_test = test_data['reviewText'].copy()
    Y_test = test_data['overall'].copy()
    
    print("Vocabulary Generation")
    
    Vocab, VocabClass, VocabSize, VocabClassSize, phi = Vocab_generation(X_train, Y_train)
    
    correct, Y_pred = random_guessing(Y_test)
    print("Random Prediction Accuracy (Test Set) = {}%".format(round(100*float(correct)/float(len(X_test)),2)))
    
    correct, Y_pred = majority_prediction(Y_test,phi)
    print("Majority Prediction Accuracy (Test Set) = {}%".format(round(100*float(correct)/float(len(X_test)),2)))
    
elif PART == 'c':
    print('Part C')
    
    train_data = preprocessing(load_data(train_path), False)
    test_data  = preprocessing(load_data(test_path), False) # False implies not to perform stemming

    X_train = train_data['reviewText'].copy()
    Y_train = train_data['overall'].copy()

    X_test = test_data['reviewText'].copy()
    Y_test = test_data['overall'].copy()
    
    print("Vocabulary Generation")
    
    Vocab, VocabClass, VocabSize, VocabClassSize, phi = Vocab_generation(X_train, Y_train)
    
    correct, Y_pred_test = predict(X_test, Y_test, phi, VocabClass, VocabClassSize)
    
    confusion_matrix = np.zeros((5,5))
    for i in range(len(Y_test)):
        confusion_matrix[Y_test[i]-1][Y_pred_test[i]-1] += 1
    
    print("Confusion Matrix")
    print(confusion_matrix.astype(int))

elif PART == 'd':
    print('Part D')
    
    train_data = preprocessing(load_data(train_path), True)
    test_data  = preprocessing(load_data(test_path), True)

    X_train = train_data['reviewText'].copy()
    Y_train = train_data['overall'].copy()

    X_test = test_data['reviewText'].copy()
    Y_test = test_data['overall'].copy()
    
    print("Vocabulary Generation")
    
    Vocab, VocabClass, VocabSize, VocabClassSize, phi = Vocab_generation(X_train, Y_train)
    
    for i in range(len(VocabClass)):
        for x in VocabClass[i]:
            VocabClass[i][x] = np.log(float(VocabClass[i][x])/float(VocabClassSize[i] + VocabSize))

    for i in range(len(VocabClass)):
        phi[i] = np.log(float(phi[i])/float(len(X_train)))
        
    correct, Y_pred_train = predict(X_train, Y_train, phi, VocabClass, VocabClassSize)
    print("Train Set Accuracy (stemmed data) = {}%".format(round(100*float(correct)/float(len(X_train)),5)))

    correct, Y_pred_test = predict(X_test, Y_test, phi, VocabClass, VocabClassSize)
    print("Test Set Accuracy (stemmed data)  = {}%".format(round(100*float(correct)/float(len(X_test)),5)))

elif PART == 'e':
    
    print('Part E (Feature Engineering)')
    print("First feature: Bigram")
    
    train_data = preprocessing(load_data(train_path), True)
    test_data  = preprocessing(load_data(test_path), True)

    X_train = train_data['reviewText'].copy()
    Y_train = train_data['overall'].copy()

    X_test = test_data['reviewText'].copy()
    Y_test = test_data['overall'].copy()
    
    X_train_b = bigrams(X_train.copy())
    X_test_b  = bigrams(X_test.copy())
    
    print("Vocabulary Generation")
    
    Vocab, VocabClass, VocabSize, VocabClassSize, phi = Vocab_generation(X_train_b, Y_train)
    
    for i in range(len(VocabClass)):
        for x in VocabClass[i]:
            VocabClass[i][x] = np.log(float(VocabClass[i][x])/float(VocabClassSize[i] + VocabSize))

    for i in range(len(VocabClass)):
        phi[i] = np.log(float(phi[i])/float(len(X_train)))
    
    correct, Y_pred_train = predict(X_train_b, Y_train, phi, VocabClass, VocabClassSize)
    print("Train Set Accuracy (bigram data) = {}%".format(round(100*float(correct)/float(len(X_train_b)),5)))

    correct, Y_pred_test = predict(X_test_b, Y_test, phi, VocabClass, VocabClassSize)
    print("Test Set Accuracy (bigram data)  = {}%".format(round(100*float(correct)/float(len(X_test_b)),5)))
    
    f1 = f1_score(np.array(Y_pred_test),Y_test,average=None)
    f1_macro = f1_score(np.array(Y_pred_test),Y_test,average='macro')
    
    print("Best model: Bigram")
    print("F1-score")
    print("Class 1:{}\nClass 2:{}\nClass 3:{}\nClass 4:{}\nClass 5:{}".format(round(f1[0],6),round(f1[1],6),round(f1[2],6),
                                                                              round(f1[3],6),round(f1[4],6)))
    print("\nMacro F1-Score: {}".format(round(f1_macro,6)))
    
    
    
    #Skip-gram
    print("\n\nSecond Feature: Skip-gram")
    X_train_s = gen_skipgrams(X_train.copy())
    X_test_s  = gen_skipgrams(X_test.copy())
    
    print("Vocabulary Generation")
    
    Vocab, VocabClass, VocabSize, VocabClassSize, phi = Vocab_generation(X_train_s, Y_train)
    
    for i in range(len(VocabClass)):
        for x in VocabClass[i]:
            VocabClass[i][x] = np.log(float(VocabClass[i][x])/float(VocabClassSize[i] + VocabSize))

    for i in range(len(VocabClass)):
        phi[i] = np.log(float(phi[i])/float(len(X_train)))
        
    correct1, Y_pred_train = predict(X_train_s, Y_train, phi, VocabClass, VocabClassSize)
    print("Train Set Accuracy (skip-gram data) = {}%".format(round(100*float(correct1)/float(len(X_train_s)),5)))

    correct2, Y_pred_test = predict(X_test_s, Y_test, phi, VocabClass, VocabClassSize)
    print("Test Set Accuracy (skip-gram data)  = {}%".format(round(100*float(correct2)/float(len(X_test_s)),5)))
    
elif PART == 'f':
    print("Part F: Summary Field")
    
    train_data = preprocessing(load_data(train_path), True)
    test_data  = preprocessing(load_data(test_path), True)
    
    X_train_summary = train_data['summary'].copy()
    Y_train_summary = train_data['overall'].copy()

    X_test_summary = test_data['summary'].copy()
    Y_test_summary = test_data['overall'].copy()
    
    print("Vocabulary Generation")
    
    Vocab, VocabClass, VocabSize, VocabClassSize, phi = Vocab_generation(X_train_summary, Y_train_summary)
    
    for i in range(len(VocabClass)):
        for x in VocabClass[i]:
            VocabClass[i][x] = np.log(float(VocabClass[i][x])/float(VocabClassSize[i] + VocabSize))

    for i in range(len(VocabClass)):
        phi[i] = np.log(float(phi[i])/float(len(X_train_summary)))
        
    correct, Y_pred_train = predict(X_train_summary, Y_train_summary, phi, VocabClass, VocabClassSize)
    print("Train Set Accuracy (stemmed data) = {}%".format(round(100*float(correct)/float(len(X_train_summary)),5)))

    correct, Y_pred_test = predict(X_test_summary, Y_test_summary, phi, VocabClass, VocabClassSize)
    print("Test Set Accuracy (stemmed data)  = {}%".format(round(100*float(correct)/float(len(X_test_summary)),5)))
    
    
    
    
    print('\n\nSummary Field with Bigrams')
    X_train_bi = gen_skipgrams(X_train_summary.copy())
    X_test_bi  = gen_skipgrams(X_test_summary.copy())
    
    print("Vocabulary Generation")
    
    Vocab, VocabClass, VocabSize, VocabClassSize, phi = Vocab_generation(X_train_bi, Y_train_summary)
    for i in range(len(VocabClass)):
        for x in VocabClass[i]:
            VocabClass[i][x] = np.log(float(VocabClass[i][x])/float(VocabClassSize[i] + VocabSize))

    for i in range(len(VocabClass)):
        phi[i] = np.log(float(phi[i])/float(len(X_train_bi)))
        
    correct, Y_pred_train_bi = predict(X_train_bi, Y_train_summary, phi, VocabClass, VocabClassSize)
    print("Train Set Accuracy (summary + bigram) = {}%".format(round(100*float(correct)/float(len(X_train_bi)),5)))

    correct, Y_pred_test_bi = predict(X_test_bi, Y_test_summary, phi, VocabClass, VocabClassSize)
    print("Test Set Accuracy (summary + bigram)  = {}%".format(round(100*float(correct)/float(len(X_test_bi)),5)))
