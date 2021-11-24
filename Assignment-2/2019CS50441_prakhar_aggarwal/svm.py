import os
import time
import sys
import warnings

sys.path.append("/home/prakank/anaconda3/lib/python3.8/site-packages/")

import scipy
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from python.svmutil import *

def load_data(filename, Binary):
    data = np.genfromtxt(filename,delimiter=',')
    data_x = data[:,:784]/255
    data_y = data[:,784]
    data_y = data_y.reshape((data_y.shape[0],1))
    
    if Binary:
        data_x = data_x[(data_y==LAST_DIGIT).reshape(-1) | ( data_y==((LAST_DIGIT+1)%10) ).reshape(-1)]
        data_y = data_y[(data_y==LAST_DIGIT).reshape(-1) | ( data_y==((LAST_DIGIT+1)%10) ).reshape(-1)]
        data_y = -1.0*(data_y==LAST_DIGIT) + 1.0*(data_y==((LAST_DIGIT+1)%10))
        
    return data_x,data_y

def svm_load_data(X,Y,d1,d2):
    X_data = X[(Y == d1).reshape(-1) | (Y == d2).reshape(-1)]
    Y_data = Y[(Y == d1).reshape(-1) | (Y == d2).reshape(-1)]
    Y_data = -1.0*(Y_data == d1) + 1.0*(Y_data == d2)
    return X_data, Y_data

def linear_kernel(X,y):
    mat = np.array(X*y)
    return np.dot(mat, mat.T)

def gaussian_kernel_element(X1,X2,gamma=0.05):
    return np.exp(-(np.linalg.norm(X1-X2)**2) * gamma)

def gaussian_kernel(X,gamma=0.05):
    X_sq   = np.sum(np.multiply(X, X),axis=1, keepdims=True)
    kernel_partial = X_sq + X_sq.T
    kernel_partial = kernel_partial - 2*np.dot(X,X.T)
    kernel = np.power(np.exp(-gamma),kernel_partial)
    return kernel

    kernel = np.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            kernel[i,j] = gaussian_kernel_element(X[i],X[j],gamma)
    return Kernel

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class SVM:
    def __init__(self,kernel,C,threshold=1e-5,gamma=0.05):
        if kernel == "linear":
            self.kernel = linear_kernel
        else:
            self.kernel = gaussian_kernel
            
        self.C = float(C)
        self.threshold = threshold
        self.gamma = gamma
    
    def train(self, X_train, Y_train):
        # minimizing function
        P = 0
        if self.kernel == linear_kernel:
            P = self.kernel(X_train,Y_train)
        elif self.kernel == gaussian_kernel:
            kernel = matrix(self.kernel(X_train, self.gamma))
            P = (kernel*Y_train)*(Y_train.T)

        P = matrix(.5 * (P + P.T))  # Just to be on the safe side (ensuring P is symmetric)
        q = matrix(-1.0*np.ones((X_train.shape[0],1)))
        c = 0.0
        
        # Inequalities
        pos = np.diag(np.ones(X_train.shape[0]))
        neg = np.diag(-np.ones(X_train.shape[0]))
        G   = matrix( np.vstack((neg,pos)) )
        
        zer   = np.zeros((X_train.shape[0]))
        c_val = self.C*np.ones(X_train.shape[0])
        h     = matrix(np.concatenate((zer,c_val)))
        
        # Equality
        A = matrix(Y_train.reshape((1,Y_train.shape[0])))
        b = matrix(0.0)
        
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b);
        
        self.alpha_whole = np.array(sol['x']).copy()
        alpha = np.array(sol['x']).copy()        
        
        self.support_vector_flag = (alpha > self.threshold).reshape(-1)
        # self.support_vector_indices = (np.arange(len(alpha)))[self.support_vector_flag]
        self.alpha = alpha[self.support_vector_flag]
        self.support_vector_x = X_train[self.support_vector_flag]
        self.support_vector_y = Y_train[self.support_vector_flag]
        
        # return self.support_vector_x, self.support_vector_y, self.alpha
        
        if self.kernel == linear_kernel:
            w_partial = self.support_vector_x * self.support_vector_y
            self.w = np.sum(w_partial * self.alpha, axis=0)
            
            b1 = np.min(X_train[(Y_train == 1).reshape(-1)] * self.w)
            b2 = np.max(X_train[(Y_train ==-1).reshape(-1)] * self.w)
            self.b  = (b1+b2)*(-0.5)
        else:
            self.w = None
            
            b1 = float("inf")
            b2 = -float("inf")
            
            kernel = gaussian_kernel(self.support_vector_x,self.gamma)
            
            for i in range(len(self.alpha)):
                kernel_partial = (kernel[i,:]).reshape(-1)
                val = ((self.alpha).reshape(-1)) * ((self.support_vector_y).reshape(-1)) * kernel_partial
                val = np.sum(val)
                # for j in range(len(self.alpha)):
                #     val += self.alpha[j]*self.support_vector_y[j]*kernel[i][j]
                if self.support_vector_y[i] == 1:
                    b1 = min(b1, val)
                else:
                    b2 = max(b2,val)
            
            if b1 == float("inf"):
                b1 = 0
            if b2 == -float("inf"):
                b2 = 0                
            self.b = (b1+b2)*(-0.5)

        return self.alpha, self.w, self.b
    
    def predict2(self,X_train,Y_train,X_test,Y_test):
        prediction = np.asmatrix(np.ones((X_test.shape[0],1),dtype=int))      
        raveled = np.asmatrix(self.alpha_whole)
        
        print(raveled.shape, self.alpha_whole.shape)
        
        Xtrain = np.sum(np.multiply(X_train,X_train),axis=1)
        Xtest  = np.sum(np.multiply(X_test,X_test),axis=1)
        X_train_X_test = np.dot(X_train,X_test.T)

        alpha_x_label = np.multiply(Y_train,np.multiply(raveled,raveled>self.threshold))
        langrangian_params = np.nonzero(raveled>self.threshold)[0]

        if len(langrangian_params)==0:
            print("No support vectors found for tolerance value= " + str(self.threshold))
        else:
            b = 0
            for sv_idx in langrangian_params:
                b+=(Y_train[sv_idx,0] - np.sum(np.multiply(alpha_x_label,np.exp(-1*self.gamma*np.sum(np.multiply(X_train-X_train[sv_idx,:],X_train-X_train[sv_idx,:]),axis=1)))))
            b = b/(float(len(langrangian_params)))
            
            for i in range(X_test.shape[0]):
                prediction[i,0] = np.sign(np.sum(np.multiply(alpha_x_label,np.exp(-1*self.gamma*(Xtrain - 2*X_train_X_test[:,i] + Xtest[i,0])))) + b)

        return prediction    

    def predict(self, X_test):   
        if self.kernel == linear_kernel:
            Y_pred = (np.dot(X_test,self.w)) + self.b
        else:
            Y_pred = np.zeros((X_test.shape[0]))
            alpha  = self.alpha.reshape(-1)
            support_vector_y = self.support_vector_y.reshape(-1)

            # kernel = 

            for i in range(X_test.shape[0]):
                mat = np.array(list(map(lambda x: gaussian_kernel_element(x,X_test[i],gamma=self.gamma),self.support_vector_x)))
                Y_pred[i] = np.sum(mat*alpha*support_vector_y) + self.b                      

        Y_pred = Y_pred.reshape(-1)
        Y_pred = np.array(list(map(lambda x: -1 if x<0 else 1,Y_pred)))
        return Y_pred

def one_vs_one_classifier_train(max_val_, X_train, Y_train, X_test, Y_test):
    svm_models = {}
    max_val = max_val_
    for i in range(max_val):
        for j in range(i+1,max(2,max_val)):
            start = time.time()
            print("\nTraining Phase: {" + str(i) + ", " + str(j) + "}")
            
            separator = "_"
            index = str(i) + separator + str(j)
            svm_models[index] = {}
            svm_models[index]["X_train"], svm_models[index]["Y_train"] = svm_load_data(X_train,Y_train,i,j)
            svm_models[index]["svm_gaussian"] = SVM(kernel="gaussian",C=1)
            svm_models[index]["params"] = svm_models[index]["svm_gaussian"].train(svm_models[index]["X_train"], svm_models[index]["Y_train"])

            end = time.time()
            print("Time: {}s".format(round(end-start,5)))

            svm_models[index]["time"] = round(end-start,5)
            
    return svm_models

def one_vs_one_classifier_pred(max_val, pred_type, svm_models, X, Y):
    y_pred_temp = np.zeros((X.shape[0], 10))
    print()
    for i in range(max_val):
        for j in range(i+1,max(2,max_val)):
            start = time.time()
            print("Prediction Phase (" + pred_type + "): {" + str(i) + ", " + str(j) + "}",end="  ")
            
            separator = "_"
            index = str(i) + separator + str(j)
            pred_string = pred_type + "_predicted"
            
            svm_models[index][pred_string] = y_pred = svm_models[index]["svm_gaussian"].predict(X)
            for k in range(y_pred.shape[0]):
                if y_pred[k] == -1:
                    y_pred_temp[k][j] += 1
                else:
                    y_pred_temp[k][i] += 1
            
            end = time.time()
            print("Time: {}s".format(round(end-start,5)))
            
            svm_models[index]["time"] += round(end-start,5)
            
    y_pred_final = [0]*y_pred_temp.shape[0]
    
    for i in range(len(y_pred_final)):
        y_pred_final[i] = 9-np.argmax((y_pred_temp[i,:])[::-1])
    
    svm_models[pred_type + "_predicted"] = np.array(y_pred_final)
    return svm_models
    
def one_vs_one_classifier(max_val_, X_train, Y_train, X_test, Y_test):
    max_val = max_val_
    
    start = time.time()
    
    svm_models = one_vs_one_classifier_train(max_val, X_train, Y_train, X_test, Y_test)
    # return svm_models

    svm_models = one_vs_one_classifier_pred (max_val, "test", svm_models, X_test, Y_test)
    # svm_models = one_vs_one_classifier_pred (max_val, "val", svm_models, X_train, Y_train)
    
    end = time.time()
    svm_models["time"] = round(end-start,5)
    
    print("Test Accuracy (Gaussian Kernel): {}%".format(round(100*accuracy_score(svm_models["test_predicted"], Y_test), 5)))
    # print("Train set Accuracy (Gaussian Kernel): {}%".format(round(100*accuracy_score(svm_models["val_predicted"], Y_train), 5)))
    
    return svm_models

BINARY_CLASSIFICATION = sys.argv[3]

if BINARY_CLASSIFICATION == '0':
    BINARY_CLASSIFICATION = True
else:
    BINARY_CLASSIFICATION = False

LAST_DIGIT = 1
BASE_DIR = "../"

# train_path = os.path.join(BASE_DIR, "data", "mnist","train.csv")
# test_path  = os.path.join(BASE_DIR, "data", "mnist","test.csv")
train_path = sys.argv[1]
test_path = sys.argv[2]

PART = sys.argv[4]

if BINARY_CLASSIFICATION:            
    if PART == 'a':
        print("Loading Data ...")
        X_train, Y_train = load_data(train_path, BINARY_CLASSIFICATION)
        X_test, Y_test  = load_data(test_path, BINARY_CLASSIFICATION)
        print("Binary classification\n\n")
        print("Training ...")
        svm_lin = SVM(kernel="linear",C=1)
        alpha,w,b = svm_lin.train(X_train,Y_train)

        print("Number of Support Vectors: {}".format(len(alpha)))
        print("W -> ",w.shape,"\n")
        print("b:",b)
        
        print("\nPredicting ...")
        Y_validation = svm_lin.predict(X_train)
        print("Validation Accuracy: {}%".format(round(100*accuracy_score(Y_validation,Y_train),3)))

        Y_pred = svm_lin.predict(X_test)
        print("Test Set Accuracy (Linear Kernel): {}%".format(round(100*accuracy_score(Y_pred,Y_test),3)))

    elif PART == 'b':
        print("Loading Data ...")
        X_train, Y_train = load_data(train_path, BINARY_CLASSIFICATION)
        X_test, Y_test  = load_data(test_path, BINARY_CLASSIFICATION)
        print("Binary classification\n\n")
        print("Training ...")
        svm_gau = SVM(kernel="gaussian",C=1)
        alpha,w,b = svm_gau.train(X_train,Y_train)

        print("Number of Support Vectors: {}".format(len(alpha)))
        print("b:",b)
        
        print("\nPredicting ...")
        Y_validation = svm_gau.predict(X_train)
        print("Validation Accuracy: {}%".format(round(100*accuracy_score(Y_validation,Y_train),3)))

        Y_pred = svm_gau.predict(X_test)
        print("Test Set Accuracy (Gaussian Kernel): {}%".format(round(100*accuracy_score(Y_pred,Y_test),3)))
        
    elif PART == 'c':
        print("Loading Data ...")
        X_train, Y_train = load_data(train_path, BINARY_CLASSIFICATION)
        X_test, Y_test  = load_data(test_path, BINARY_CLASSIFICATION)
        print("Binary classification\n\n")
        print("Training (Linear Kernel)...")
        model1 = svm_train(Y_train.reshape(-1), X_train, '-s 0 -c 1 -t 0')
        
        print("\nPredicting ...")
        label_predict, accuracy,  decision_values  =svm_predict(Y_test.reshape(-1),X_test,model1, '-q');
        label_predict2, accuracy2, decision_values2=svm_predict(Y_train.reshape(-1),X_train,model1, '-q');
                
        print("{}Linear Score (Test set): {}% {}".format(color.BOLD,round(100*accuracy_score(label_predict,Y_test),3),color.END ) )
        print("{}Linear Score (Train set): {}% {}".format(color.BOLD,round(100*accuracy_score(label_predict2,Y_train),3),color.END ) )
        
        
        
        print("\n\nTraining (Gaussian Kernel)...")
        model2 = svm_train(Y_train.reshape(-1), X_train, '-s 0 -c 1 -t 2 -g 0.05')
        
        print("\nPredicting ...")
        label_predict_g, accuracy_g, decision_values_g   =svm_predict(Y_test.reshape(-1),X_test,model2,'-q');
        label_predict_g2, accuracy_g2, decision_values_g2=svm_predict(Y_train.reshape(-1),X_train,model2, '-q');
        
        print("{}RBF Score (Test set): {}% {}".format(color.BOLD,round(100*accuracy_score(label_predict_g,Y_test),3),color.END ) )
        print("{}RBF Score (Train set): {}% {}".format(color.BOLD,round(100*accuracy_score(label_predict_g2,Y_train),3),color.END ) )
    
    else:
        print("Invalid part number")

else:
    print("Loading Data ... ")
    if PART == 'a':
        X_train, Y_train = load_data(train_path, False)
        X_test, Y_test  = load_data(test_path, False)
        
        max_val = 3 # Max class to include (Max-1 will be included)
        
        start = time.time()
        print("Training ...")
        svm_models = one_vs_one_classifier(max_val, X_train, Y_train, X_test, Y_test)    
        
        print("Predicting ...")
        y_pred_train, y_pred_test = svm_models["val_predicted"], svm_models["test_predicted"]
        
        end = time.time()
        print("\n\nGlobal Time: {}".format(round(end-start,7)))
    
    elif PART == 'b':
        
        X_train, Y_train = load_data(train_path, False)
        X_test, Y_test  = load_data(test_path, False)
        
        start = time.time()
        print("Training ...")
        model = svm_train(Y_train.reshape(-1), X_train, '-s 0 -c 1 -t 2 -g 0.05 -q')
        
        print("Predicting ...")
        
        label_predict1, accuracy, decision_values=svm_predict(Y_train.reshape(-1),X_train,model,'-q')
        print("Train Accuracy :", accuracy_score(label_predict1, Y_train))

        # confusion_matrix1 = np.zeros((10,10))
        # for i in range(Y_train.shape[0]):
        #     confusion_matrix1[int(Y_train[i])][int(label_predict1[i])] += 1
        # print("Training set")
        # print(confusion_matrix1.astype(int))


        label_predict2, accuracy, decision_values=svm_predict(Y_test.reshape(-1),X_test,model, '-q')
        print("Test Accuracy :", accuracy_score(label_predict2, Y_test))
        
        # confusion_matrix2 = np.zeros((10,10))
        # for i in range(Y_test.shape[0]):
        #     confusion_matrix2[int(Y_test[i])][int(label_predict2[i])] += 1
        # print("Test set")
        # print(confusion_matrix2.astype(int))
        
        end = time.time()
        print("\n\nGlobal Time: {}".format(round(end-start,7)))        
    
    
    elif PART == 'c':
        
        X_train, Y_train = load_data(train_path, False)
        X_test, Y_test  = load_data(test_path, False)
        
        max_val = 10 # Max class to include (Max-1 will be included)
        
        start = time.time()
        print("Training ...")
        svm_models = one_vs_one_classifier(max_val, X_train, Y_train, X_test, Y_test)    
        
        print("Predicting ...")
        svm_models["val_predicted"] = []
        y_pred_train, y_pred_test = svm_models["val_predicted"], svm_models["test_predicted"]
        
        end = time.time()
        print("\n\nGlobal Time: {}".format(round(end-start,7)))
        confusion_matrix0 = np.zeros((10,10))
        for i in range(Y_test.shape[0]):
            confusion_matrix0[int(Y_test[i])][int(y_pred_test[i])] += 1
        print("Test set")
        print(confusion_matrix0.astype(int))
        
        sys.exit()
        start = time.time()
        print("Training ...") 
        model = svm_train(Y_train.reshape(-1), X_train, '-s 0 -c 1 -t 2 -g 0.05 -q')
        
        print("Predicting ...")
        
        label_predict1, accuracy, decision_values=svm_predict(Y_train.reshape(-1),X_train,model,'-q')
        print("Train Accuracy :", accuracy_score(label_predict1, Y_train))

        confusion_matrix1 = np.zeros((10,10))
        for i in range(Y_train.shape[0]):
            confusion_matrix1[int(Y_train[i])][int(label_predict1[i])] += 1
        print("Training set")
        print(confusion_matrix1.astype(int))


        label_predict2, accuracy, decision_values=svm_predict(Y_test.reshape(-1),X_test,model, '-q')
        print("Test Accuracy :", accuracy_score(label_predict2, Y_test))
        
        confusion_matrix2 = np.zeros((10,10))
        for i in range(Y_test.shape[0]):
            confusion_matrix2[int(Y_test[i])][int(label_predict2[i])] += 1
        print("Test set")
        print(confusion_matrix2.astype(int))
        
        end = time.time()
        print("\n\nGlobal Time: {}".format(round(end-start,7)))
        
    elif PART == 'd':
        
        X_train, Y_train = load_data(train_path, False)
        X_test, Y_test  = load_data(test_path, False)
                
        C_list = [1e-5, 1e-3, 1, 5, 10]

        acc_train = []
        acc_pred  = []
        time_list = []

        for C in C_list:
            print("\nValue of C: " + str(C))
            start = time.time()
            
            x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.2)
            command = '-s 0 -t 2 -g 0.05 -c ' + str(C) + ' -q'
            model = svm_train(y_train.reshape(-1), x_train, command)
            
            y_pred, accuracy, dec_values = svm_predict(y_val.reshape(-1), x_val, model, '-q')
            temp_acc = accuracy_score(y_pred, y_val)
            print("Validation Set (Dev set): {}".format(temp_acc))
            acc_train.append(temp_acc)

            
            y_pred, accuracy, dec_values = svm_predict(Y_test.reshape(-1), X_test, model, '-q')
            temp_acc = accuracy_score(y_pred, Y_test)
            print("Test Set (Dev set): {}".format(temp_acc))
            acc_pred.append(temp_acc)
            
            end = time.time()
            
            time_list.append(round(end-start,5))
            print("Time:",round(time_list[-1],5))

        y1 = [0.09275, 0.0935, 0.97175, 0.97325, 0.97125]
        y2 = [0.1009, 0.1009, 0.9705, 0.9715, 0.9706]
        x = [1e-5, 1e-3, 1, 5, 10]

        x  = np.array(x)
        y1 = np.array(y1)
        y2 = np.array(y2)

        plt.title('5-fold cross-validation accuracy and Test Accuracy')
        plt.xlabel('log(C)')
        plt.ylabel('Accuracy')
        plt.scatter(np.log10(x),y1,color='red',label='Cross-validation')
        plt.scatter(np.log10(x),y2,color='blue',label='Test accuracy')
        plt.legend()
        plt.savefig('Cross-validation accuracy and Test Accuracy.jpg')
        plt.show()
    
    else:
        print("Invalid part number")