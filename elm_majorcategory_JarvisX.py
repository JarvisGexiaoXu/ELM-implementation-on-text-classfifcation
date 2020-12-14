# extreme learning machine
# Jarvis Xu

import numpy as np
import pandas as pd # mainly used for reading data from excel
import nltk # mainly used for eliminating the stop words
from nltk.corpus import stopwords # ^
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score

# used to transform text data into 
from sklearn.feature_extraction.text import HashingVectorizer 

# preprocessing
# X_train, y_train
# X_test, y_test

# load data from excel
def load(s):
    data = pd.read_excel('EMAILDATASET.xlsx')
    lst = list(data[s])
    return lst

# eliminate stopword from a sentence
def removeStopWords(s,all_stopwords):
    text_tokens = nltk.word_tokenize(s)
    tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]
    newString = " ".join(word for word in tokens_without_sw)
    return newString

# convert text input into numbers
def X_convert(lst):
    # eliminate stopwords
    nltk.download('stopwords')
    all_stopwords = stopwords.words('english')
    X = []
    num_inputNodes = 2000 # this number controls the number of nodes in the input layer
    vectorizer = HashingVectorizer(n_features = num_inputNodes)
    print("The number of input nodes:", num_inputNodes)
    for i in range(len(lst)):
        text = lst[i]
        text = [removeStopWords(text, all_stopwords)]
        vector = vectorizer.transform(text)
        X.append((vector.toarray())[0])
    return X

# convert text desired output into numbers 
def desiredOutput(lst):
    output = []
    for i in lst:
        if i == 'ITD': output.append(0)
        elif i == 'OAA': output.append(1)
        elif i == 'OSL': output.append(2)
        else: output.append(0)
    return output

# extract the test data from the complete set of data
def testSet(lst):
    test = []
    for i in range(240,255):
        test.append(lst[i])
    return test
'''
# print the actual results
def print_Result(D,A):
    D1 = 0
    A1 = 0
    for i in range(len(D)):
        print()
'''
# ELM neural network implementation
class ELM():
    # define input X, label y, number of neurons m, 
    # contral parameter L = 0.2 and training function TRAIN_beta
    def __init__(self, X, y, m, L):
        self.X = X
        self.y = y
        self.m = m
        self.L = L
        self.TRAIN_beta()
    
    # use sigmoid function for feature mapping
    # transform input data into ELM feature space
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    
    # define training function, random w, b
    # output matrix H, input weights beta
    # F1 output function
    def TRAIN_beta(self):
        n, d = self.X.shape
        self.w = np.random.rand(d, self.m)
        self.b = np.random.rand(1, self.m)
        H = self.sigmoid(np.dot(self.X,self.w) + self.b) # use feature mapping to get output matrix
        self.beta = np.dot(np.linalg.inv(np.identity(self.m) / self.L + np.dot(H.T, H)),
                    np.dot(H.T, self.y)) # β = inv(H)·T
        print('Train Finish', self.beta.shape,"(# hidden nodes, # output nodes)")

# testing function
    def TEST(self, x):
        H = self.sigmoid(np.dot(x, self.w) + self.b)  # use testing set
        result = np.dot(H, self.beta) # T = H·β
        # print('result= ',result)
        return result

X_name = 'Text_Of_Email'
y_name = 'Category'
X = load(X_name)
y = load(y_name)

# preprocessing for input data
X_train = X_convert(X)
# preprocessing for desired output
y_train = desiredOutput(y)

X_test = testSet(X_train)
y_test = testSet(y_train)   

X_train = X_train[0:241]
y_train = y_train[0:241]
X_test1 = X_train[100:114]
y_test1 = y_train[100:114]


X_train = np.array(X_train)
y_train = np.array(y_train)
X_train1 = np.array(X_train)
y_train1 = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_test1 = np.array(X_test)
y_test1 = np.array(y_test)

# training process
# OneHot encode is used for training
Y_onehot = np.eye(3)[y_train]
elm = ELM(X_train, Y_onehot, 5000, 0.2)#-------------------------------------------------------------

'''
# testing process
predict = elm.TEST(X_test)
predict = np.argmax(predict, axis = 1) # use OneHot encode, classify by the index with the greatest value
y_test = np.eye(3)[y_test]
acc = np.sum(predict == y_test)
print('acc :', acc)
for i in range(len(y_test)):
    print(y_test[i])
    print(predict[i])
'''

# testing process over training set
predict = elm.TEST(X_train)
predict = np.argmax(predict, axis = 1) # use OneHot encode, classify by the index with the greatest value
y_train1 = np.eye(3)[y_train]
c_m = confusion_matrix(y_train,predict)
print("confusion matrix for training set:")
print(c_m)
print("precision score = ",precision_score(y_train,predict,average='micro'))
print("recall score = ",recall_score(y_train,predict,average='micro'))
print("f1 score = ",f1_score(y_train,predict,average='micro'))

# testing process over testing set
predict = elm.TEST(X_test1)
predict = np.argmax(predict, axis = 1) # use OneHot encode, classify by the index with the greatest value
y_test1 = np.eye(3)[y_test1]
# Confusion matrix for testing set
c_m = confusion_matrix(y_test,predict)
print("confusion matrix for testing set:")
print(c_m)
print("precision score = ",precision_score(y_test,predict,average='micro'))
print("recall score = ",recall_score(y_test,predict,average='micro'))
print("f1 score = ",f1_score(y_test,predict,average='micro'))
# print(y_test)
# print(predict)

'''
# now the raw data is ready to be used-----------------------------
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
'''















