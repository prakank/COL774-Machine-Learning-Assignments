import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

learning_rate = 0.01
DELTA = 1e-8
BATCH_SIZE = 10
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K_ITERATIONS = 10

# global EXAMPLES

def h_theta(theta, x):
    if type(x) == int or type(x) == float:
        x = np.array([x,1])
    return np.dot(np.transpose(theta), x)

def loss_function(y, theta, x):
    temp = (y - h_theta(theta, x))
    return temp*temp

def data_load(filename):
    path = os.path.join(BASE_DIR, 'data', 'q1', filename)
    data = np.genfromtxt(path, delimiter=',')
    return data

def normalize(data):
    data = (data - data.mean())/data.std()
    return data

def gradient_loss_function(data_X, data_Y, theta): # -(1/batch_size) * summation_1_batch_size ( ( y(i) - h_theta(x_i) ) * x(i) )
    gradient = np.zeros((2))
    loss = 0
    for i in range(BATCH_SIZE):
        index = np.random.randint(100)
        x = np.array((data_X[index], 1.0))
        partial_compute = ((data_Y[index] - h_theta(theta, x))*x).reshape(x.shape[0])
        partial_loss    = loss_function(data_Y[index], theta, x)
        
        gradient = np.add(gradient,partial_compute)
        loss     = partial_loss + loss
        
    gradient = np.multiply(gradient,-1/BATCH_SIZE)
    loss = (loss)/(BATCH_SIZE*2)
    
    return loss, gradient

def converge(loss1, loss2):
    return abs(loss1 - loss2) <= DELTA 

def check(theta, data_X, data_Y):
    for i in range(EXAMPLES):
        x = np.array((data_X[i], 1.0))
        # print(h_theta(theta, x), data_Y[i])

def graph_loss(data):
    plt.plot(data)
    y = np.arange(len(data))
    plt.title("Loss function (SGD) vs Iteration")
    plt.xlabel("Iteration Number")
    plt.ylabel("Loss function (SGD)")
    plt.savefig("output.jpg")
    # plt.xticks(y)    
    plt.show()

def graph_y_yhat(data_x, data_y, theta):
    
    min_x = float('inf')
    min_x_idx = -1
    max_x = -float('inf')
    max_x_idx = -1
    
    for i in range(len(data_x)):
        if(data_x[i] < min_x):
            min_x = data_x[i]
            min_x_idx = i
            
        if(data_x[i] > max_x):
            max_x = data_x[i]
            max_x_idx = i
    
    min_x = math.floor(data_x[min_x_idx])
    max_x = math.ceil(data_x[max_x_idx])
    
    plt.scatter(data_x,data_y, label='Original Value')
    plt.plot([min_x, max_x], [h_theta(theta, min_x), h_theta(theta, max_x)], label='Predicted Value', color='orange')
    plt.title("Acidity vs Density")
    plt.xlabel("Acidity (Normalized)")
    plt.ylabel("Density")
    plt.savefig("y_yhat.jpg")
    plt.legend()
    plt.show()
    
def linear_regression(data_x, data_y):
    data_x = normalize(data_x)
    if data_x.shape[0] != data_y.shape[0]:
        raise IOError("Inconsistent data")
    global EXAMPLES
    EXAMPLES = data_x.shape[0]
    theta = np.zeros((2)) # Initialized theta to be a zero vector
    
    count1, count2 = 0, 0
    loss1, loss2 = 0, 0
    i = 0
    
    data_plot = []
    
    while True:
        i+=1
        partial_loss, gradient = gradient_loss_function(data_x, data_y, theta)
        theta = np.add(theta, np.multiply(gradient, -learning_rate))
        
        data_plot.append(partial_loss)
        
        if count1 < K_ITERATIONS:
            count1+=1
            loss1 += partial_loss
        elif  count2 < K_ITERATIONS:
            count2+=1
            loss2 += partial_loss
            
        if(count2 == K_ITERATIONS):
            loss1 /= BATCH_SIZE
            loss2 /= BATCH_SIZE
            count1 = 0
            count2 = 0
        
        if converge(loss1,loss2):
            # print("Iterations: {}".format(i))
            print("Parameters Learned: {}".format(theta))
            check(theta, data_x, data_y)
            # graph_loss(data_plot)
            graph_y_yhat(data_x, data_y, theta)
            return

if __name__ == '__main__':
    X = data_load('linearX.csv')
    Y = data_load('linearY.csv')
    linear_regression(X,Y)