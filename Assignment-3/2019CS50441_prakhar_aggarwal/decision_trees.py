import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import math
import time
import re
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = ''

train_path = os.path.join(BASE_DIR, 'data', 'bank_dataset', 'bank_train.csv')
test_path  = os.path.join(BASE_DIR, 'data', 'bank_dataset', 'bank_test.csv')
val_path   = os.path.join(BASE_DIR, 'data', 'bank_dataset', 'bank_val.csv')

train_path = sys.argv[1]
test_path = sys.argv[2]
val_path = sys.argv[3]

PART = sys.argv[4]

def load_data(one_hot_encoding, numeric_cols, filename, values_dict = {}):
    df = pd.read_csv(filename, delimiter = ';')
    Y = df['y'].copy()
    Y  = Y.to_numpy()
    for i in range(Y.shape[0]):
        if Y[i] == 'yes':
            Y[i] = 1
        else:
            Y[i] = 0 #Assigning 0 to nan values
    
    Y = Y.astype('int64')
    df = df.drop(['y'],axis=1)
    
    if one_hot_encoding == False:
        return df, Y, {}
    
    if one_hot_encoding == True and values_dict == {}:
        for col in df.columns:
            if col not in numeric_cols:
                values = list(set(list(df[col])))
                values_dict[col] = values
                
                for i in range(df.shape[0]):
                    df[col][i] = values_dict[col].index(df[col][i])
        return df, Y, values_dict
    
    elif one_hot_encoding == True:
        for col in df.columns:
            if col not in numeric_cols:
                for i in range(df.shape[0]):
                    if df[col][i] in values_dict[col]:
                        df[col][i] = values_dict[col].index(df[col][i])
                    else:
                        df[col][i] = -1
        return df, Y, {}
    
def entropy_num(X,Y):
    median = np.median(X)
    boolean_flag = X > median
    # X_left  = X[boolean_flag == False]
    # X_right = X[boolean_flag == True]
    Y_left  = Y[boolean_flag == False]
    Y_right = Y[boolean_flag == True]
    
    p1,p2 = 0,0
    
    if Y_left.shape[0] > 0:
        s1 = np.sum(Y_left)
        s1 = max(s1, Y_left.shape[0] - s1)
        p1 = float(s1)/float(Y_left.shape[0])
        p1 = - p1 * np.log(p1)
        
        if Y_left.shape[0] > s1:
            p_t = float(Y_left.shape[0]-s1)/float(Y_left.shape[0])
            p_t = -p_t * np.log(p_t)        
            p1 += p_t
        
        p1 = p1*(float(Y_left.shape[0]))/float(Y.shape[0])
    
    if Y_right.shape[0] > 0:
        s2 = np.sum(Y_right)
        s2 = max(s2, Y_right.shape[0] - s2)
        p2 = float(s2)/float(Y_right.shape[0])
        p2 = - p2 * np.log(p2)
        
        if Y_right.shape[0] > s2:
            p_t = float(Y_right.shape[0]-s2)/float(Y_right.shape[0])
            p_t = -p_t * np.log(p_t)
            p2 += p_t
        
        p2 = p2*(float(Y_right.shape[0]))/float(Y_right.shape[0])
    
    # print(p1,p2,X,Y)
    
    return p1+p2

def entropy_categorical(X,Y):
    val = list(set(list(X)))
    val_count = dict.fromkeys(val,[0,0])
    
    for i in range(X.shape[0]):
        val_count[X[i]][1] += 1
        if Y[i] == 1:
            val_count[X[i]][0] += 1
    entr = 0

    for category,count in val_count.items():
        p = 0
        val_count[category][0] = max(count[0], count[1] - count[0])
        if count[1] > 0:
            p = float(val_count[category][0])/float(count[1])
            p = -p * np.log(p)

            p_t = float(count[1]-val_count[category][0])/float(count[1])
            p_t = -p_t * np.log(p_t)

            p += p_t
            
            p = p * (float(count[1]))/float(Y.shape[0])        
        entr += p
    
    return entr
    
def information_gain(attribute, one_hot_encoding, numeric_cols, parent_entr, indices, X, Y):
    X_new = np.array((X.iloc[indices])[attribute])
    Y_new = Y[indices]
    entr  = 0
    info_parent = parent_entr
    
    if attribute in numeric_cols:
        entr = entropy_num(X_new, Y_new)
        # print("Info Gain:",attribute,entr)
    else:
        if one_hot_encoding == False: # Multi split
            entr = entropy_categorical(X_new, Y_new)
        else:
            entr = entropy_num(X_new, Y_new)
    
    
    
    return info_parent - entr

def best_attribute(one_hot_encoding, rem_attr, numeric_cols, parent, indices, X, Y):
    best_attr = ''
    info_gain = -float('inf')
    
    parent_entr = 0
    # if parent == None:
    #     parent_entr = 0
    # elif parent.attr in numeric_cols:
    #     parent_entr = entropy_num
    
    for attr in X.columns:
        if attr in numeric_cols or attr in rem_attr:            
            temp = information_gain(attr, one_hot_encoding, numeric_cols, parent_entr, indices, X, Y)
            # print("Best_Attr Selection:",attr,temp)
            if temp > info_gain:
                info_gain = temp
                best_attr = attr
    
    return best_attr, info_gain

class dc_node:
    
    # indices coming at this node
    def __init__(self,parent,indices,depth,decision,median=0,value=None,attribute=None):
        self.parent = parent
        self.indices = indices

        self.child = []
        self.depth = depth
        self.attr  = attribute

        self.decision = decision
        self.median   = median
        self.value    = value
   
def construct_decision_tree(one_hot_encoding, rem_attr, numeric_cols, parent, indices, X, Y, MAX_DEPTH):
    Y_new = np.array(Y[indices])
    if np.sum(Y_new) > (Y_new.shape[0] - np.sum(Y_new)):
        decision = 1
    else:
        decision = 0
    
    if indices.shape[0] == 1 or (parent != None and parent.depth >= MAX_DEPTH):
        if parent == None:
            return dc_node(parent, indices, 1, decision)
        else:
            return dc_node(parent, indices, parent.depth+1, decision)
    else:
        attr,gain = best_attribute(one_hot_encoding, rem_attr, numeric_cols, parent, indices, X, Y)
        
        depth = 0
        if parent == None:
            depth = 0
        else:
            depth = parent.depth + 1
        node = dc_node(parent,indices,depth,decision,attribute=attr)
        
        # print("Attr:",attr,",  Gain: ",gain,",  Depth:",depth)
        # print("Rem:",rem_attr)
        # print("Numeric:",numeric_cols)
        # print("Indices:",indices)
        
        # if gain > 0:
        X_new = np.array((X.iloc[indices])[attr])
        
        if attr in numeric_cols or (one_hot_encoding == True):
            median = np.median(X_new)
            node.median = median
            
            boolean_flag = X_new > median
            ind_left  = indices[boolean_flag == False]
            ind_right = indices[boolean_flag == True]
            
            if one_hot_encoding == True and attr not in numeric_cols:
                boolean_flag = (X_new % 2 == 0)
                ind_left  = indices[boolean_flag == False]
                ind_right = indices[boolean_flag == True]
                rem_attr.remove(attr)
                
            
            # print("Left: ",ind_left,ind_left.shape[0])
            # print("Right: ",ind_right,ind_right.shape[0])
            # print("Indices: ", indices, indices.shape[0])

            if ind_left.shape[0] > 0:
                left  = construct_decision_tree(one_hot_encoding, rem_attr.copy(), numeric_cols, node, ind_left, X, Y, MAX_DEPTH)
                left.value = 'left'
                node.child.append(left)
            
            if ind_right.shape[0] > 0:
                right = construct_decision_tree(one_hot_encoding, rem_attr.copy(), numeric_cols, node, ind_right, X, Y, MAX_DEPTH)
                right.value = 'right'
                node.child.append(right)
                                    
        elif one_hot_encoding == False and attr in rem_attr:
            rem_attr.remove(attr)
            val = list(set(list(X_new)))
            for i in val:
                ind = indices[X_new == i]
                if ind.shape[0] > 0:
                    child = construct_decision_tree(one_hot_encoding, rem_attr.copy(), numeric_cols, node, ind, X, Y, MAX_DEPTH)
                    child.value = i
                    node.child.append(child)
        return node

def decision_tree(one_hot_encoding, categorical_cols, numeric_cols, X, Y, MAX_DEPTH = 20):
    indices = np.arange(0,X.shape[0])
    dc_tree = construct_decision_tree(one_hot_encoding, categorical_cols.copy(), numeric_cols, None, indices, X, Y, MAX_DEPTH)
    return dc_tree

def predict_recursive(one_hot_encoding, x, root, numeric_cols):
    if len(root.child) == 0:
        return root.decision
    else:
        val = ''
        if root.attr in numeric_cols:
            if x[root.attr] > root.median:
                val = 'right'
            else:
                val = 'left'
        elif one_hot_encoding == True:
            index = x[root.attr]
            if index % 2 == 0:
                val = 'right'
            else:
                val = 'left'
        else:            
            val = x[root.attr]
        
        for i in range(len(root.child)):
            if root.child[i].value == val:
                return predict_recursive(one_hot_encoding, x, root.child[i], numeric_cols)
        return root.decision
    
def predict(one_hot_encoding, X, root, numeric_cols):
    Y_pred = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
    # for i in range(1):
        z = predict_recursive(one_hot_encoding, X.iloc[i], root, numeric_cols)
        Y_pred[i] = int(z)
    Y_pred = Y_pred.astype('int64')
    return Y_pred

def print_tree(root):
    if root.parent==None:
        print("Root Node. Feature Used to Split-> " + str(root.attr))
        print("Decision: " + str(root.decision))
        child_list = []
        for i in range(len(root.child)):
            child_list.append(root.child[i].value)
            
        print("Child Values -> " + str(child_list))

        for c in root.child:
            print_tree(c)
    else:
        print("Child splitted on feature -> " + str(root.parent.attr))
        print("Decision: " + str(root.decision))
        print("Value of child is -> " + str(root.value))
        # print("Feature to be used -> " + str(root.attr))
        child_list = []
        for i in range(len(root.child)):
            child_list.append(root.child[i].value)
            
        print("Child Values -> " + str(child_list))

        for c in root.child:
            print_tree(c)        
        
def accuracy(y1,y2):
    count = 0.0
    for i in range(y1.shape[0]):
        if y1[i] == y2[i]:
            count+=1.0
    return (count)/(y1.shape[0])

def bfs(root):
    i = 0
    node_list = [root]
    while i < len(node_list):
        top_node = node_list[i]
        if len(top_node.child) > 0:
            for c in top_node.child:
                node_list.append(c)
        i+=1
    return node_list        

def load_data_rf(numeric_cols, filename, cols_train, values_dict = {}):
        df = pd.read_csv(filename, delimiter = ';')
        Y = df['y'].copy()
        Y  = Y.to_numpy()
        for i in range(Y.shape[0]):
            if Y[i] == 'yes':
                Y[i] = 1
            else:
                Y[i] = 0 #Assigning 0 to nan values
        
        Y = Y.astype('int64')
        df = df.drop(['y'],axis=1)
        
        x = {}
        
        
        if values_dict == {}:
            cols = []
            for col in df.columns:
                if col not in numeric_cols:
                    values = sorted(list(set(list(df[col]))))
                    values_dict[col] = values
                    new_cols = [col+str(i) for i in range(len(values))]
                    cols = cols + new_cols
                else:
                    cols.append(col)
        else:
            cols = cols_train
            
            
        for i in cols:
            x[i] = np.zeros((df.shape[0]))
        
        for i in range(df.shape[0]):
            for j in df.columns:
                if j in numeric_cols:
                    x[j][i] = df[j][i]
                elif df[j][i] in values_dict[j]:
                    x[j+str(values_dict[j].index(df[j][i]))][i] = 1
        
        df_new = pd.DataFrame.from_dict(x)
        
        return df_new, Y, cols, values_dict

        if values_dict == {}:
            for col in df.columns:
                if col not in numeric_cols:
                    values = list(set(list(df[col])))
                    values_dict[col] = values                
                    for i in range(df.shape[0]):
                        temp = df[col][i]
                        df[col][i] = np.zeros(len(values))
                        df[col][i][values_dict[col].index(temp)] = 1
            return df, Y, values_dict
        
        else:
            for col in df.columns:
                if col not in numeric_cols:
                    for i in range(df.shape[0]):
                        temp = df[col][i]
                        df[col][i] = np.zeros(len(values_dict[col]))
                        if temp in values_dict[col]:
                            df[col][i][values_dict[col].index(temp)] = 1                        
            return df, Y    


if PART == 'a':
    one_hot_encoding = False
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    print("Loading data...")
    Xtrain, Ytrain, _ = load_data(one_hot_encoding, numeric_cols, train_path)

    print("Loading test data...")
    Xtest, Ytest, _ = load_data(one_hot_encoding, numeric_cols, test_path)

    print("Loading val data...")
    Xval, Yval, _ = load_data(one_hot_encoding, numeric_cols, val_path)
    
    prediction = {}

    MAX_DEPTH = 20

    for depth in range(MAX_DEPTH+1):
        
        print("Current Depth:",depth)
        
        temp = []
        
        print("Training ...")
        start = time.time()    
        dc_tree = decision_tree(one_hot_encoding, categorical_cols, numeric_cols, Xtrain, Ytrain, depth)    
        end = time.time()
        
        temp.append(round(end-start,5))
        start = time.time()

        print("Prediction ...")
        y_pred = predict(one_hot_encoding, Xtrain, dc_tree, numeric_cols)
        acc1 = round(accuracy(y_pred,Ytrain),3)
        print("Train Accuracy: {}%".format(100.0*acc1))

        y_pred2 = predict(one_hot_encoding, Xtest, dc_tree, numeric_cols)
        acc2 = round(accuracy(y_pred2,Ytest),3)
        print("Test Accuracy: {}%".format(100.0*acc2))

        y_pred3 = predict(one_hot_encoding, Xval, dc_tree, numeric_cols)
        acc3 = round(accuracy(y_pred3,Yval),3)
        print("Val Accuracy: {}%\n\n".format(100.0*acc3))
        
        end = time.time()
        temp.append(round(end-start,5))
                
        temp.extend([acc1, acc2, acc3])
        
        node_list = bfs(dc_tree)
        temp.insert(0,len(node_list))
        
        prediction[str(depth)] = temp
        
    train_acc = []
    test_acc = []
    val_acc = []
    total_nodes = []
    time_train = []
    time_pred = []

    for i in prediction:
        total_nodes.append(prediction[i][0])
        time_train.append(prediction[i][1])
        time_pred.append(prediction[i][2])
        train_acc.append(prediction[i][3])
        test_acc.append(prediction[i][4])
        val_acc.append(prediction[i][5])
        
    def graph_plot(train, test, val, total_nodes, filename):        
        plt.figure(figsize=(10, 6))
        
        x = [int(i) for i in range(len(train))]
        
        plt.title("Train/Test/Val Accuracies vs Depth/Nodes")
        plt.xlabel("Depth")
        plt.ylabel("Accuracies")
        plt.plot(x, train,label="Train")
        plt.plot(x, test,label="Test")
        plt.plot(x, val, label="Val")
        plt.legend()
        plt.savefig(filename)
        
        plt.show()

    if one_hot_encoding == True:
        graph_plot(train_acc, test_acc, val_acc, total_nodes, "output/decision_trees_accuracy_vs_depth_ohe.jpg")
    else:
        graph_plot(train_acc, test_acc, val_acc, total_nodes, "output/decision_trees_accuracy_vs_depth.jpg")
        
if PART == 'b':
    one_hot_encoding = True
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    print("Loading data...")
    Xtrain, Ytrain, values_dict = load_data(one_hot_encoding, numeric_cols, train_path, {})

    print("Loading test data...")
    Xtest, Ytest, _ = load_data(one_hot_encoding, numeric_cols, test_path, values_dict)

    print("Loading val data...")
    Xval, Yval, _ = load_data(one_hot_encoding, numeric_cols, val_path, values_dict)
    
    print("Building Decision Tree ... ")
    dc_tree = decision_tree(one_hot_encoding, categorical_cols, numeric_cols, Xtrain, Ytrain, 9)
    
    node_list = bfs(dc_tree)
    train_accuracy = []
    test_accuracy = []
    val_accuracy = []
    node_count = [len(node_list)]

    train_accuracy.append(accuracy_score(predict(one_hot_encoding, Xtrain, dc_tree, numeric_cols), Ytrain))
    test_accuracy.append(accuracy_score(predict(one_hot_encoding, Xtest, dc_tree, numeric_cols), Ytest))
    val_accuracy.append(accuracy_score(predict(one_hot_encoding, Xval, dc_tree, numeric_cols), Yval))
    
    iteration = 0
    val_best_accuracy = val_accuracy[-1]

    while iteration < 5:
        iteration += 1
        previous_accuracy = val_accuracy[-1]
        after_accuracy    = None
        tree_best_node    = tree
        
        count = 0
        for node in node_list:
            count = count + 1
            if len(node.child) > 0:
                child_temp = node.child
                node.child = []
                after_accuracy = accuracy_score(predict(one_hot_encoding, Xval, dc_tree, numeric_cols), Yval)
                
                if count%10 == 0:
                    print("After: ",after_accuracy,"Val Best: ",val_best_accuracy,"Node number:",count)
                    
                if after_accuracy > val_best_accuracy:
                    val_best_accuracy = after_accuracy
                    tree_best_node = node
                    # break
                node.child = child_temp
        
        print("Best Accuracy: ",val_best_accuracy,"  Previous: ",previous_accuracy)
        if val_best_accuracy > previous_accuracy:
            tree_best_node.child = [] # PRUNING
        
            node_list = bfs(dc_tree)
            train_accuracy.append(accuracy_score(predict(one_hot_encoding, Xtrain, dc_tree, numeric_cols), Ytrain))
            test_accuracy.append(accuracy_score(predict(one_hot_encoding, Xtest, dc_tree, numeric_cols), Ytest))
            val_accuracy.append(accuracy_score(predict(one_hot_encoding, Xval, dc_tree, numeric_cols), Yval))
            node_count.append(len(node_list))

            print("Iteration: {}, Val_accuracy: {}, Node_count: {}".format(iteration, val_accuracy[-1], node_count[-1]))        
        else:
            break
        
    print(node_count)
    print("Train: ", train_accuracy)
    print("Test: ", test_accuracy)
    print("Val: ", val_accuracy)
    
    def plot_part_b(train, test, val, node_count):
        plt.plot(node_count, train, label='Train')
        plt.plot(node_count, test, label='Test')
        plt.plot(node_count, val, label='Val')
        plt.xlabel('Node Count')
        plt.ylabel('Accuracy')
        plt.title("Accuracy vs Node Count (Pruning)")
        plt.legend()
        plt.savefig('output/part_b_accuracy_node_pruning.jpg')
        plt.show()
    
    plot_part_b(train_accuracy, test_accuracy, val_accuracy, node_count)
    
if PART == 'c':
    
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    print("Loading train data...")
    Xtrain_rf, Ytrain_rf, cols, values_dict_rf = load_data_rf(numeric_cols, train_path, [], {})

    print("Loading test data...")
    Xtest_rf, Ytest_rf, _, _ = load_data_rf(numeric_cols, test_path, cols, values_dict_rf)

    print("Loading val data...")
    Xval_rf, Yval_rf, _, _ = load_data_rf(numeric_cols, val_path, cols, values_dict_rf)

    param_grid = {
        'n_estimators': [50, 150, 250, 350, 450],
        'max_features': [0.1, 0.3, 0.5, 0.7, 0.9, 1],
        'min_samples_split': [2, 4, 6, 8, 10],
        'max_depth': [4, 8, 12, 16, 20]
    }

    best_params = {
        'n_estimators': [350],
        'max_features': [0.3],
        'min_samples_split': [10],
        'max_depth': [20]
    }
    
    start_time = time.time()
    print("Training ...")
    param_grid = best_params
    rfc = RandomForestClassifier(criterion='entropy', bootstrap=True, oob_score=True)
    classifier = GridSearchCV(rfc, param_grid)
    classifier.fit(Xtrain_rf, Ytrain_rf)
    end_time = time.time()
    
    print("Training Time:{}s".format(round(end_time - start_time,5)))
    print("Best Params:{}".format(classifier.best_params_))
    
    
    print("Prediction ...")                    
    y_pred_train = classifier.predict(Xtrain_rf)
    y_pred_acc_train = accuracy_score(y_pred_train, Ytrain_rf)
    print("Training Accuracy:{}%".format(round(100*y_pred_acc_train,5)))

    y_pred_test = classifier.predict(Xtest_rf)
    y_pred_acc_test = accuracy_score(y_pred_test, Ytest_rf)
    print("Test Accuracy:{}%".format(round(100*y_pred_acc_test,5)))

    y_pred_val = classifier.predict(Xval_rf)
    y_pred_acc_val = accuracy_score(y_pred_val, Yval_rf)
    print("Validation Accuracy:{}%\n\n".format(round(100*y_pred_acc_val,5)))                
    # print("Out of bag error:{}".format(rfc.oob_score_))
    
    # start_time = time.time()
    # # param_grid = best_params
    # models = {}

    # best_acc = -1
    # optimal_param = {}

    # for ne in param_grid['n_estimators']:
    #     for mf in param_grid['max_features']:
    #         for mss in param_grid['min_samples_split']:
    #             for md in param_grid['max_depth']:
    #                 index = "{}_{}_{}_{}".format(ne,mf,mss,md)
    #                 models[index] = {}
    #                 start_time_ = time.time()
    #                 print("Params:\nn_estimators:{}, max_features:{}, min_samples_split:{}, max_depth:{}".format(ne, mf, mss, md))
    #                 print("Training ...")
    #                 rfc = RandomForestClassifier(criterion='entropy', bootstrap=True, oob_score=True, random_state=1,
    #                                             n_estimators=ne, max_features=mf, min_samples_split=mss, max_depth=md)
    #                 rfc = rfc.fit(Xtrain_rf, Ytrain_rf)
                    
    #                 end_time_ = time.time()
    #                 print("Training Time:{}s".format(round(end_time_ - start_time_,5)))
    #                 models[index]['time'] = round(end_time_ - start_time_,5)
                    
    #                 print("Prediction ...")
                    
    #                 y_pred_train = rfc.predict(Xtrain_rf)
    #                 y_pred_acc_train = accuracy_score(y_pred_train, Ytrain_rf)
    #                 print("Training Accuracy:{}%".format(round(100*y_pred_acc_train,5)))

    #                 y_pred_test = rfc.predict(Xtest_rf)
    #                 y_pred_acc_test = accuracy_score(y_pred_test, Ytest_rf)
    #                 print("Test Accuracy:{}%".format(round(100*y_pred_acc_test,5)))

    #                 y_pred_val = rfc.predict(Xval_rf)
    #                 y_pred_acc_val = accuracy_score(y_pred_val, Yval_rf)
    #                 print("Validation Accuracy:{}%\n\n".format(round(100*y_pred_acc_val,5)))
                    
    #                 models[index]['train_accuracy'] = y_pred_acc_train
    #                 models[index]['test_accuracy'] = y_pred_acc_test
    #                 models[index]['val_accuracy'] = y_pred_acc_val
    #                 models[index]['error'] = rfc.oob_score_
                    
    #                 if y_pred_acc_test > best_acc:
    #                     best_acc = y_pred_acc_test
    #                     optimal_param['n_estimators'] = ne
    #                     optimal_param['max_featues'] = mf
    #                     optimal_param['min_samples_split'] = mss
    #                     optimal_param['max_depth'] = md
                        
                                    
    # end_time = time.time()
    # print("Total Time: ", end_time - start_time)

if PART == 'd':
    # Part D (Parameter Sensitivity Analysis)

    accuracies_dict = {}
    
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    print("Loading train data...")
    Xtrain_rf, Ytrain_rf, cols, values_dict_rf = load_data_rf(numeric_cols, train_path, [], {})

    print("Loading test data...")
    Xtest_rf, Ytest_rf, _, _ = load_data_rf(numeric_cols, test_path, cols, values_dict_rf)

    print("Loading val data...")
    Xval_rf, Yval_rf, _, _ = load_data_rf(numeric_cols, val_path, cols, values_dict_rf)

    param_grid = {
        'n_estimators': [50, 150, 250, 350, 450],
        'max_features': [0.1, 0.3, 0.5, 0.7, 0.9, 1],
        'min_samples_split': [2, 4, 6, 8, 10],
        'max_depth': [4, 8, 12, 16, 20]
    }

    best_params = {
        'n_estimators': [350],
        'max_features': [0.3],
        'min_samples_split': [10],
        'max_depth': [20]
    }

    for i in param_grid:
        current_param = {}
        for j in best_params:
            if j != i:
                current_param[j] = best_params[j][0]
        
        accuracies_dict[i] = {'train':[], 'test':[], 'val':[]}
        
        for val in param_grid[i]:
            current_param[i] = val
            
            print("Params:{}".format(current_param))
            print("Training ....")        
            rfc = RandomForestClassifier(n_estimators=current_param['n_estimators'], oob_score=True,bootstrap=True,
                                        max_features=current_param['max_features'],max_depth=current_param['max_depth'],
                                        min_samples_split=current_param['min_samples_split'],criterion='entropy',random_state=1)
            rfc = rfc.fit(Xtrain_rf, Ytrain_rf)
            
            y_pred_train = rfc.predict(Xtrain_rf)
            y_pred_acc_train = accuracy_score(y_pred_train, Ytrain_rf)
            print("Training Accuracy:{}%".format(round(100*y_pred_acc_train,5)))

            y_pred_test = rfc.predict(Xtest_rf)
            y_pred_acc_test = accuracy_score(y_pred_test, Ytest_rf)
            print("Test Accuracy:{}%".format(round(100*y_pred_acc_test,5)))

            y_pred_val = rfc.predict(Xval_rf)
            y_pred_acc_val = accuracy_score(y_pred_val, Yval_rf)
            print("Validation Accuracy:{}%".format(round(100*y_pred_acc_val,5)))
            print("Out of bag error:{}%\n".format(rfc.oob_score_))

            accuracies_dict[i]['train'].append(y_pred_acc_train)
            accuracies_dict[i]['test'].append(y_pred_acc_test)
            accuracies_dict[i]['val'].append(y_pred_acc_val)
    
    def graph_plot_rf(accuracies_dict, param_grid):
        for i in param_grid:
            # Plot accuracies_dict[i]
            filename = "output/part_d_accuracy_vs_" + i + ".jpg"
            y1 = accuracies_dict[i]['train']
            y2 = accuracies_dict[i]['test']
            y3 = accuracies_dict[i]['val']
            
            x = param_grid[i]
            
            # plt.figure(figsize=(10, 6))
            
            plt.title("Train/Test/Val Accuracy vs " + i)
            plt.xlabel(i)
            plt.ylabel("Accuracies")
            plt.plot(x,y1,label="Train")
            plt.plot(x,y2,label="Test")
            plt.plot(x,y3, label="Val")
            
            plt.legend()
            plt.savefig(filename)
            
            plt.show()

    graph_plot_rf(accuracies_dict, param_grid)