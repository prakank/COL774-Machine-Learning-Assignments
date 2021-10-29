import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import sys
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')

def load_data(filename, num_of_features, num_of_target_class, values_dict = {}, train_data = False):
    df = np.asarray(pd.read_csv(filename, header=None, dtype=int))
    df = df[df[:,-1] < num_of_target_class]
    
    Y = df[:,-1]
    df = df[:,0:df.shape[1]-1]
    
    y = np.zeros((df.shape[0],len(list(set(list(Y))))))
    
    df = df[:, 0:num_of_features]

    count = 0
    if values_dict == {}:
        for i in range(df.shape[1]):
            length = len(list(set(list(df[:,i]))))
            values_dict[i] = length
            count += length
        x = np.zeros((df.shape[0],count))
    else:
        x = np.zeros((df.shape[0],sum(values_dict.values())))
    
    for j in range(df.shape[0]):
        ohe_encoded = []
        y[j][Y[j]] = 1
        for i in range(df.shape[1]):
            val = df[j][i]
            ohe_mat = np.zeros((values_dict[i]))
            ohe_mat[val-1] = 1
            ohe_encoded.extend(ohe_mat)
        ohe_encoded = np.asarray(ohe_encoded)
        x[j] = ohe_encoded

    if train_data == True:
        return x,y,values_dict
    else:
        return x,y

class NN_architecture:
    def __init__(self, learning_rate, batch_size, num_of_features, hidden_layer_units, num_of_outputs, MIN_LOSS, epsilon, adaptive_learning = False):
        self.learning_rate = learning_rate
        self.init_learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_of_features = num_of_features
        self.hidden_layer_list = hidden_layer_units
        self.num_of_outputs = num_of_outputs
        self.MIN_LOSS = MIN_LOSS
        self.epsilon = epsilon
        self.learning_rate_threshold = 1e-5 # Min possible learning rate
        self.adaptive = adaptive_learning
    
    def sigmoid_activation(self,data):
        return (1.0/(1.0 + np.exp(-data)) )
    
    def relu_activation(self,data):
        return np.multiply(data>0,data)
    
    def sigmoid_der(self,data):
        # x = self.sigmoid_activation(data)
        x = data
        return np.multiply(x,1.0-x)
    
    def relu_der(self,data):
        temp = np.ones(data.shape,dtype=float)
        return np.multiply(temp>0,temp)
    
    def initialize(self):
        neuron_count = [self.num_of_inputs]
        neuron_count.extend(self.hidden_layer_list)
        neuron_count.append(self.num_of_outputs)
        self.neuron_count = neuron_count
        params = {}
        np.random.seed(1)
        
        # xavier initialization
        for i in range(1, len(neuron_count)):
            params["W" + str(i)] = np.random.normal(0,1,(neuron_count[i],neuron_count[i-1]))*np.sqrt(2.0/neuron_count[i-1])
            params["b" + str(i)] = np.zeros((neuron_count[i],1),dtype=float)
        
        self.params = params
        self.num_of_layers = len(neuron_count)
        
        return
    
    def forward_propagation(self, X, activation_function):        
        forward_prop = {}
        data = (X.T).copy()
        forward_prop["a0"] = data # n*m

        for i in range(self.num_of_layers-2):
            data = np.dot(self.params["W"+str(i+1)], data) + self.params["b"+str(i+1)]

            if activation_function == "relu":
                data = self.relu_activation(data)
            elif activation_function == "sigmoid":
                data = self.sigmoid_activation(data)

            forward_prop["a"+str(i+1)] = data.copy()
        
        data = np.dot(self.params["W"+str(self.num_of_layers-1)], data) + self.params["b"+str(self.num_of_layers-1)]
        data = self.sigmoid_activation(data)
        forward_prop["a"+str(self.num_of_layers-1)] = data.copy()

        self.forward_prop = forward_prop
        return
    
    def backward_propagation(self, Y, activation_function):
        self.backward_prop = {}
        dataY = (Y.T).copy()
        
        self.backward_prop["dz"+str(self.num_of_layers-1)] = self.forward_prop["a" + str(self.num_of_layers-1)] - dataY
        
        for i in range(self.num_of_layers-2,0,-1):
            temp_mat = np.dot(self.params["W"+str(i+1)].T, self.backward_prop["dz"+str(i+1)])
            
            if activation_function == "sigmoid":
                temp_mat =np.multiply(temp_mat, self.sigmoid_der(self.forward_prop["a"+str(i)]))
            elif activation_function == "relu":
                temp_mat =np.multiply(temp_mat, self.relu_der(self.forward_prop["a"+str(i)]))
                
            self.backward_prop["dz"+str(i)] = temp_mat
        
        # i = self.num_of_layers-2
        # while i>=0:
        #     temp_mat = np.dot(self.params["W"+str(i+1)].T, backward_prop["dz" + str(i+1)])
            
        #     if activation_function == "sigmoid":
        #         temp_mat = np.multiply(temp_mat, self.sigmoid_der(self.forward_prop["a"+str(i)]))
        #     elif activation_function == "relu":
        #         temp_mat = np.multiply(temp_mat, self.relu_der(self.forward_prop["a"+str(i)]))
                
        #     backward_prop["dz" + str(i)] = temp_mat            
        #     i-=1
        # self.backward_prop = backward_prop
        # return
    
    def backward_propagation2(self, Y, activation_function):
        self.backward_prop = {}
        dataY = (Y.T).copy()
        
        temp_mat = np.multiply(dataY - self.forward_prop["a"+str(self.num_of_layers-1)], self.forward_prop["a"+str(self.num_of_layers-1)])
        temp_mat = np.multiply(temp_mat, (1-self.forward_prop["a"+str(self.num_of_layers-1)]) )
        
        self.backward_prop["dz"+str(self.num_of_layers-1)] = temp_mat
        
        for i in range(self.num_of_layers-2, 0, -1):
            if activation_function == "sigmoid":
                temp_mat = np.multiply(self.params["W"+str(i+1)].T @ self.backward_prop["dz"+str(i+1)], self.forward_prop["a"+str(i)] * (1-self.forward_prop["a"+str(i)]) )
            elif activation_function == "relu":
                temp_mat = temp_mat = np.multiply(self.params["W"+str(i+1)].T @ self.backward_prop["dz"+str(i+1)], np.multiply(self.forward_prop["a"+str(i)] > 0, self.forward_prop["a"+str(i)]) )
        
            self.backward_prop["dz"+str(i)] = temp_mat
        return
    
    def update_params(self,M):
        new_params = {}
        for i in range(1,self.num_of_layers):
            new_params["W"+str(i)] = self.params["W"+str(i)] - (self.learning_rate/M)*np.dot(self.backward_prop["dz"+str(i)],(self.forward_prop["a"+str(i-1)]).T)
            
            temp = (self.learning_rate/M)*np.sum(self.backward_prop["dz"+str(i)],axis=1)
            temp = temp.reshape((temp.shape[0],1))
            

            new_params["b"+str(i)] = self.params["b"+str(i)] - temp

        self.params = new_params
        return
    
    def predict(self,X,activation_function="sigmoid"):
        data_x = (X.T).copy()
        for i in range(1,self.num_of_layers):
            data_x = np.add(np.dot(self.params["W"+str(i)],data_x),self.params["b"+str(i)])
            if activation_function == "sigmoid":
                data_x = self.sigmoid_activation(data_x)
                
            elif activation_function == "relu":
                data_x = self.relu_activation(data_x)    
        
        data_x = self.sigmoid_activation(data_x)
        data_x = data_x.T
        data_x = data_x/(np.sum(data_x,axis=1).reshape(data_x.shape[0],1))
                
        return data_x, np.argmax(data_x,axis=1)
    
    def loss_function(self,y1,y2):
        # print(y1,y2)
        y = np.abs(y1-y2)
        y = np.multiply(y,y)
        return np.sum(y)/(2*y.shape[0])
    
    def print_param(self, i):
        print("Iteration: {}".format(i))
        for i in self.params:
            print(i,np.max(self.params[i]), np.min(self.params[i]), self.params[i].shape)
        print()

        for i in self.backward_prop:
            print(i,np.max(self.backward_prop[i]), np.min(self.backward_prop[i]), self.backward_prop[i].shape)
        print()    

        for i in self.forward_prop:
            print(i,np.max(self.forward_prop[i]), np.min(self.forward_prop[i]), self.forward_prop[i].shape)
        print()
    
    def print_class_param(self):
        print("Batch Size: {}, Learning rate: {}, Num of layers: {}".format(self.batch_size, self.learning_rate, self.num_of_layers))
        print("Neuron count: {}".format(self.neuron_count))
    
    def entropy_loss(self, y1, y2):
        
        # Here y2 is the actual Y
        # y1 is the predicted y
        l1 = np.log(np.multiply(1, y1==0) + y1)
        l1 = np.multiply(l1, y2)
        
        l2 = np.log(np.multiply(1, y1==1) + 1 - y1)
        l2 = np.multiply(l2, 1-y2)
        
        l = np.mean(-1.0*l1-1.0*l2,axis=1)
        l = np.sqrt(np.sum(np.multiply(l,l)))/(2.0*y1.shape[1])
        
        return l
    
    def run(self,epochs,X,Y,activation_function):
        self.num_of_inputs = X.shape[1]
        self.examples = X.shape[0]
        self.batches = (int)(self.examples/self.batch_size)        
        self.initialize()
        # self.print_param(0)
        # self.print_class_param()
        
        iteration  = 1
        error = float("inf")        
        time_start = time.time()
        print("Training phase ... ")
        
        error_list = []
        
        while iteration <= epochs and error > self.epsilon and self.learning_rate > self.learning_rate_threshold:
            
            error = 0
            for batch in range(self.batches):
                start = batch*self.batch_size
                end   = min(start + self.batch_size,self.examples)

                X_new = X[start:end,:]
                Y_new = Y[start:end,:]

                self.forward_propagation(X_new,activation_function)

                # self.backward_propagation2(Y_new,activation_function)
                self.backward_propagation(Y_new,activation_function)

                self.update_params(Y_new.shape[0])

                # loss_partial = self.entropy_loss(self.forward_prop["a"+str(self.num_of_layers-1)], Y_new.T)
                loss_partial = self.loss_function(self.forward_prop["a"+str(self.num_of_layers-1)], Y_new.T)
                
                error += (loss_partial)
            
            error = error/self.batches
            error_list.append(error)
            
            # Convergence criteria
            if len(error_list) > 1:
                if error_list[-1] <= self.MIN_LOSS and error_list[-2] <= self.MIN_LOSS and (abs(error_list[-1] - error_list[-2])) < (self.epsilon):
                    time_end = time.time()
                    self.training_time = (time_end - time_start)
                    return

            if iteration%200 == 0:
                print("Epoch: {}, Error: {}, Learning Rate: {}".format(iteration, error, self.learning_rate))
            iteration += 1

            if self.adaptive:
                self.learning_rate = (self.init_learning_rate)/(np.sqrt(iteration))

        time_end = time.time()
        self.training_time = (time_end - time_start)

# Drawing the Confusion matrix and saving in filename
def draw_conf_matrix(x,units, filename):
    plt.imshow(x)
    plt.title("Confusion Matrix (" + str(units) + " units)")
    plt.colorbar()
    plt.set_cmap("Greens")
    plt.ylabel("True labels")
    plt.xlabel("Predicted label")
    plt.savefig(filename)
    plt.show()

PART = sys.argv[3]
BASE_DIR = '../'

# train_path = os.path.join(BASE_DIR, 'data', 'Poker_Hand_dataset', 'poker-hand-training-true.data')
train_path = sys.argv[1]

# test_path  = os.path.join(BASE_DIR, 'data', 'Poker_Hand_dataset', 'poker-hand-testing.data')
test_path = sys.argv[2]

BATCH_SIZE = 100
NUMBER_OF_FEATURES = 10
HIDDEN_LAYERS_UNITS = [5]
NUMBER_OF_TARGET_CLASS = 10
NUMBER_OF_OUTPUTS = NUMBER_OF_TARGET_CLASS

LEARNING_RATE = 0.1
EPSILON = 0.0001
MIN_LOSS = 0.01
tolerance = 0.001
EPOCHS = 100

# def run_predict_plot(hidden_layer, epochs, activation_function, adaptive, Xtrain, Ytrain):
#     temp = []
#     model = NN_architecture(LEARNING_RATE, BATCH_SIZE, NUMBER_OF_FEATURES, hidden_layer, NUMBER_OF_OUTPUTS, MIN_LOSS, EPSILON, adaptive)
#     model.run(epochs,Xtrain, Ytrain,activation_function)
#     t = round(model.training_time,3)
#     print("Training time: {}s".format(t))
    
#     y_class_train, y_pred_train = model.predict(Xtrain,activation_function)
#     error = model.loss_function(y_class_train, Ytrain)
#     train_a = round(100*accuracy_score(y_pred_train, np.argmax(Ytrain,axis=1)),3)
#     print("Train Accuracy: {}%, Error: {}".format(train_a, error))

#     y_class_test, y_pred_test = model.predict(Xtest,activation_function)
#     error2 = model.loss_function(y_class_test, Ytest)
#     test_a = round(100*accuracy_score(y_pred_test, np.argmax(Ytest,axis=1)),3)
#     print("Test Accuracy: {}%, Error: {}".format(test_a, error2))

#     y_conf = np.argmax(Ytest, axis=1)
#     confusion_matrix = np.zeros((NUMBER_OF_TARGET_CLASS,NUMBER_OF_TARGET_CLASS))
#     for i in range(Ytest.shape[0]):
#         confusion_matrix[y_conf[i]][y_pred_test[i]] += 1

#     confusion_matrix = confusion_matrix.astype(int)
#     temp = [t, train_a, test_a, confusion_matrix]
#     prediction[l[0]] = temp

#     # print(confusion_matrix)
#     draw_conf_matrix(confusion_matrix, l[0], "../output/conf_matrix_" + str(l[0]) + ".jpg")
    
    

    

if PART == 'a':
    Xtrain, Ytrain, values_dict = load_data(train_path, NUMBER_OF_FEATURES, NUMBER_OF_TARGET_CLASS, train_data=True)
    Xtest, Ytest = load_data(test_path, NUMBER_OF_FEATURES, NUMBER_OF_TARGET_CLASS, values_dict)
    
    hidden_layer = [[5], [10], [15], [20], [25]]
    prediction = {}

    for l in hidden_layer:
        temp = []
        model = NN_architecture(LEARNING_RATE, BATCH_SIZE, NUMBER_OF_FEATURES, l, NUMBER_OF_OUTPUTS, MIN_LOSS, EPSILON, False)
        model.run(1000,Xtrain, Ytrain,'sigmoid')
        t = round(model.training_time,3)
        print("Training time: {}s".format(t))
        
        y_class_train, y_pred_train = model.predict(Xtrain,'sigmoid')
        error = model.loss_function(y_class_train, Ytrain)
        train_a = round(100*accuracy_score(y_pred_train, np.argmax(Ytrain,axis=1)),3)
        print("Train Accuracy: {}%, Error: {}".format(train_a, error))

        y_class_test, y_pred_test = model.predict(Xtest,'sigmoid')
        error2 = model.loss_function(y_class_test, Ytest)
        test_a = round(100*accuracy_score(y_pred_test, np.argmax(Ytest,axis=1)),3)
        print("Test Accuracy: {}%, Error: {}".format(test_a, error2))

        y_conf = np.argmax(Ytest, axis=1)
        confusion_matrix = np.zeros((NUMBER_OF_TARGET_CLASS,NUMBER_OF_TARGET_CLASS))
        for i in range(Ytest.shape[0]):
            confusion_matrix[y_conf[i]][y_pred_test[i]] += 1

        confusion_matrix = confusion_matrix.astype(int)
        temp = [t, train_a, test_a, confusion_matrix]
        prediction[l[0]] = temp

        # print(confusion_matrix)
        draw_conf_matrix(confusion_matrix, l[0], "../output/conf_matrix_sigmoid_" + str(l[0]) + ".jpg")
        
        train_acc = []
        test_acc  = []
        time_list = []

        for i in prediction:
            time_list.append(prediction[i][0])
            train_acc.append((prediction[i][1])/100.0)
            test_acc.append((prediction[i][2])/100.0)
        
        def graph_plot(y1, y2, title, filename):
            x = [5, 10, 15, 20, 25]
            plt.figure(figsize=(10, 6))    

            plt.title(title)
            plt.xlabel("Units")
            plt.ylabel("Accuracies")
            plt.plot(x, y1,label="Train")
            plt.plot(x, y2,label="Test")
            plt.legend()
            plt.savefig(filename)    
            plt.show()

        graph_plot(train_acc, test_acc, "Train/Test Accuracies vs Hidden Units (Sigmoid, Single Layer)", "../output/nn_accuracy_vs_hidden_units_normal.jpg")
    
# ADAPTIVE LEARNING
if PART == 'd':
    Xtrain, Ytrain, values_dict = load_data(train_path, NUMBER_OF_FEATURES, NUMBER_OF_TARGET_CLASS, train_data=True)
    Xtest, Ytest = load_data(test_path, NUMBER_OF_FEATURES, NUMBER_OF_TARGET_CLASS, values_dict)    

    hidden_layer = [[5], [10], [15], [20], [25]]
    prediction = {}

    ADAPTIVE_LEARNING = True
    EPOCHS = 1000

    for l in hidden_layer:
        temp = []
        model = NN_architecture(LEARNING_RATE, BATCH_SIZE, NUMBER_OF_FEATURES, l, NUMBER_OF_OUTPUTS, MIN_LOSS, EPSILON, ADAPTIVE_LEARNING)
        model.run(EPOCHS,Xtrain, Ytrain,'sigmoid')
        t = round(model.training_time,3)
        print("Training time: {}s".format(t))
        
        y_class_train, y_pred_train = model.predict(Xtrain,'sigmoid')
        error = model.loss_function(y_class_train, Ytrain)
        train_a = round(100*accuracy_score(y_pred_train, np.argmax(Ytrain,axis=1)),3)
        print("Train Accuracy: {}%, Error: {}".format(train_a, error))

        y_class_test, y_pred_test = model.predict(Xtest,'sigmoid')
        error2 = model.loss_function(y_class_test, Ytest)
        test_a = round(100*accuracy_score(y_pred_test, np.argmax(Ytest,axis=1)),3)
        print("Test Accuracy: {}%, Error: {}".format(test_a, error2))

        y_conf = np.argmax(Ytest, axis=1)
        confusion_matrix = np.zeros((NUMBER_OF_TARGET_CLASS,NUMBER_OF_TARGET_CLASS))
        for i in range(Ytest.shape[0]):
            confusion_matrix[y_conf[i]][y_pred_test[i]] += 1

        confusion_matrix = confusion_matrix.astype(int)
        temp = [t, train_a, test_a, confusion_matrix]
        prediction[l[0]] = temp

        # print(confusion_matrix)
        draw_conf_matrix(confusion_matrix, l[0], "../output/conf_matrix_sigmoid_adaptive_" + str(l[0]) + ".jpg")
        
        train_acc = []
        test_acc  = []
        time_list = []

        for i in prediction:
            time_list.append(prediction[i][0])
            train_acc.append((prediction[i][1])/100.0)
            test_acc.append((prediction[i][2])/100.0)

        def graph_plot(y1, y2, filename):
            x = [5, 10, 15, 20, 25]
            plt.figure(figsize=(10, 6))    
            
            plt.title("Train/Test Accuracies vs Hidden Units (Sigmoid, Adaptive, Epochs: 1000)")
            plt.xlabel("Units")
            plt.ylabel("Accuracies")
            plt.plot(x, y1,label="Train")
            plt.plot(x, y2,label="Test")
            plt.legend()
            plt.savefig(filename)    
            plt.show()

        graph_plot(train_acc, test_acc, "../output/nn_accuracy_vs_hidden_units_sigmoid_adaptive_"+str(EPOCHS)+".jpg")

if PART == 'e':
    Xtrain, Ytrain, values_dict = load_data(train_path, NUMBER_OF_FEATURES, NUMBER_OF_TARGET_CLASS, train_data=True)
    Xtest, Ytest = load_data(test_path, NUMBER_OF_FEATURES, NUMBER_OF_TARGET_CLASS, values_dict)    
    
    # SIGMOID (2 Layers)
    hidden_layer = [[100, 100]]
    prediction = {}
    EPOCHS = 1000

    for l in hidden_layer:
        temp = []
        model = NN_architecture(LEARNING_RATE, BATCH_SIZE, NUMBER_OF_FEATURES, l, NUMBER_OF_OUTPUTS, MIN_LOSS, EPSILON, False)
        model.run(EPOCHS,Xtrain, Ytrain,'sigmoid')
        t = round(model.training_time,3)
        print("Training time: {}s".format(t))
        
        y_class_train, y_pred_train = model.predict(Xtrain,'sigmoid')
        error = model.loss_function(y_class_train, Ytrain)
        train_a = round(100*accuracy_score(y_pred_train, np.argmax(Ytrain,axis=1)),3)
        print("Train Accuracy: {}%, Error: {}".format(train_a, error))

        y_class_test, y_pred_test = model.predict(Xtest,'sigmoid')
        error2 = model.loss_function(y_class_test, Ytest)
        test_a = round(100*accuracy_score(y_pred_test, np.argmax(Ytest,axis=1)),3)
        print("Test Accuracy: {}%, Error: {}".format(test_a, error2))

        y_conf = np.argmax(Ytest, axis=1)
        confusion_matrix = np.zeros((NUMBER_OF_TARGET_CLASS,NUMBER_OF_TARGET_CLASS))
        for i in range(Ytest.shape[0]):
            confusion_matrix[y_conf[i]][y_pred_test[i]] += 1

        confusion_matrix = confusion_matrix.astype(int)
        temp = [t, train_a, test_a, confusion_matrix]
        prediction[str(l[0])+"_"+str(l[1])] = temp

        # print(confusion_matrix)
        draw_conf_matrix(confusion_matrix, l[0], "../output/conf_matrix_sigmoid_2_layers_" + str(l[0]) + "_" + str(l[1]) + ".jpg")




    # SIGMOID (2 Layers, Adaptive Learning Rate)
    hidden_layer = [[100, 100]]
    prediction = {}

    for l in hidden_layer:
        temp = []
        model = NN_architecture(LEARNING_RATE, BATCH_SIZE, NUMBER_OF_FEATURES, l, NUMBER_OF_OUTPUTS, MIN_LOSS, EPSILON, True)
        model.run(1000,Xtrain, Ytrain,'sigmoid')
        t = round(model.training_time,3)
        print("Training time: {}s".format(t))
        
        y_class_train, y_pred_train = model.predict(Xtrain,'sigmoid')
        error = model.loss_function(y_class_train, Ytrain)
        train_a = round(100*accuracy_score(y_pred_train, np.argmax(Ytrain,axis=1)),3)
        print("Train Accuracy: {}%, Error: {}".format(train_a, error))

        y_class_test, y_pred_test = model.predict(Xtest,'sigmoid')
        error2 = model.loss_function(y_class_test, Ytest)
        test_a = round(100*accuracy_score(y_pred_test, np.argmax(Ytest,axis=1)),3)
        print("Test Accuracy: {}%, Error: {}".format(test_a, error2))

        y_conf = np.argmax(Ytest, axis=1)
        confusion_matrix = np.zeros((NUMBER_OF_TARGET_CLASS,NUMBER_OF_TARGET_CLASS))
        for i in range(Ytest.shape[0]):
            confusion_matrix[y_conf[i]][y_pred_test[i]] += 1

        confusion_matrix = confusion_matrix.astype(int)
        temp = [t, train_a, test_a, confusion_matrix]
        prediction[str(l[0])+"_"+str(l[1])] = temp

        # print(confusion_matrix)
        draw_conf_matrix(confusion_matrix, l, "../output/conf_matrix_sigmoid_adaptive_2_layers_" + str(l[0]) + "_" + str(l[1]) + ".jpg")



    # Relu (2 Layers)
    hidden_layer = [[100, 100]]
    prediction = {}

    for l in hidden_layer:
        temp = []
        model = NN_architecture(LEARNING_RATE, BATCH_SIZE, NUMBER_OF_FEATURES, l, NUMBER_OF_OUTPUTS, MIN_LOSS, EPSILON, False)
        model.run(1000,Xtrain, Ytrain,'relu')
        t = round(model.training_time,3)
        print("Training time: {}s".format(t))
        
        y_class_train, y_pred_train = model.predict(Xtrain,'relu')
        error = model.loss_function(y_class_train, Ytrain)
        train_a = round(100*accuracy_score(y_pred_train, np.argmax(Ytrain,axis=1)),3)
        print("Train Accuracy: {}%, Error: {}".format(train_a, error))

        y_class_test, y_pred_test = model.predict(Xtest,'relu')
        error2 = model.loss_function(y_class_test, Ytest)
        test_a = round(100*accuracy_score(y_pred_test, np.argmax(Ytest,axis=1)),3)
        print("Test Accuracy: {}%, Error: {}".format(test_a, error2))

        y_conf = np.argmax(Ytest, axis=1)
        confusion_matrix = np.zeros((NUMBER_OF_TARGET_CLASS,NUMBER_OF_TARGET_CLASS))
        for i in range(Ytest.shape[0]):
            confusion_matrix[y_conf[i]][y_pred_test[i]] += 1

        confusion_matrix = confusion_matrix.astype(int)
        temp = [t, train_a, test_a, confusion_matrix]
        prediction[str(l[0])+"_"+str(l[1])] = temp

        # print(confusion_matrix)
        draw_conf_matrix(confusion_matrix, l, "../output/conf_matrix_relu_2_layers_" + str(l[0]) + "_" + str(l[1]) + ".jpg")

    
    
    
    # Relu (2 Layers, Adaptive Learning Rate)
    hidden_layer = [[100, 100]]
    prediction = {}

    for l in hidden_layer:
        temp = []
        model = NN_architecture(LEARNING_RATE, BATCH_SIZE, NUMBER_OF_FEATURES, l, NUMBER_OF_OUTPUTS, MIN_LOSS, EPSILON, True)
        model.run(1000,Xtrain, Ytrain,'relu')
        t = round(model.training_time,3)
        print("Training time: {}s".format(t))
        
        y_class_train, y_pred_train = model.predict(Xtrain,'relu')
        error = model.loss_function(y_class_train, Ytrain)
        train_a = round(100*accuracy_score(y_pred_train, np.argmax(Ytrain,axis=1)),3)
        print("Train Accuracy: {}%, Error: {}".format(train_a, error))

        y_class_test, y_pred_test = model.predict(Xtest,'relu')
        error2 = model.loss_function(y_class_test, Ytest)
        test_a = round(100*accuracy_score(y_pred_test, np.argmax(Ytest,axis=1)),3)
        print("Test Accuracy: {}%, Error: {}".format(test_a, error2))

        y_conf = np.argmax(Ytest, axis=1)
        confusion_matrix = np.zeros((NUMBER_OF_TARGET_CLASS,NUMBER_OF_TARGET_CLASS))
        for i in range(Ytest.shape[0]):
            confusion_matrix[y_conf[i]][y_pred_test[i]] += 1

        confusion_matrix = confusion_matrix.astype(int)
        temp = [t, train_a, test_a, confusion_matrix]
        prediction[str(l[0])+"_"+str(l[1])] = temp

        # print(confusion_matrix)
        draw_conf_matrix(confusion_matrix, l, "../output/conf_matrix_relu_adaptive_2_layers_" + str(l[0]) + "_" + str(l[1]) + ".jpg")



# MLPClassifier
if PART == 'e':
    start_time = time.time()
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(100,100), activation = 'relu', max_iter=1000, learning_rate='adaptive').fit(Xtrain,np.argmax(Ytrain,axis=1))
    end_time = time.time()

    print("Training time: {}s".format(round(end_time-start_time,5)))
    
    def accuracy(y1, y2):
        count = 0.0
        for i in range(y2.shape[0]):
            if y1[i] == y2[i]:
                count+=1.0
        return (count)/(float(y2.shape[0]))
    
    y_pred_train_mlp = clf.predict(Xtrain)
    train_accuracy_mlp = round(100*accuracy(y_pred_train_mlp, np.argmax(Ytrain,axis=1)),5)
    print("Train Accuracy: {}%".format(train_accuracy_mlp))
        
    y_pred_test_mlp = clf.predict(Xtest)
    test_accuracy_mlp = round(100*accuracy(y_pred_test_mlp, np.argmax(Ytest,axis=1)),5)
    print("Test Accuracy: {}%".format(test_accuracy_mlp))
    
    y_conf_MLP = np.argmax(Ytest, axis=1)
    confusion_matrix = np.zeros((NUMBER_OF_TARGET_CLASS,NUMBER_OF_TARGET_CLASS))
    for i in range(Ytest.shape[0]):
        confusion_matrix[y_conf_MLP[i]][y_pred_test_mlp[i]] += 1

    confusion_matrix = confusion_matrix.astype(int)
    draw_conf_matrix(confusion_matrix, l, "../output/conf_matrix_relu_adaptive_2_layers_100_100_MLP.jpg")
    