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

def batch_gradient(data_X, data_Y, theta): # -(1/batch_size) * summation_1_batch_size ( ( y(i) - h_theta(x_i) ) * x(i) )
    gradient = np.zeros((2))
    loss = 0
    for i in range(BATCH_SIZE):
        index = np.random.randint(100)
        x = np.array((data_X[index], 1.0))
        partial_gradient = ((data_Y[index] - h_theta(theta, x))*x).reshape(x.shape[0])
        partial_loss    = loss_function(data_Y[index], theta, x)
        
        gradient = np.add(gradient,partial_gradient)
        loss     = partial_loss + loss
        
    gradient = np.multiply(gradient,-1/BATCH_SIZE)
    loss = (loss)/(2*BATCH_SIZE)
    
    return loss, gradient

def converge(loss1, loss2):
    return abs(loss1 - loss2) <= DELTA 

def predict(theta, data_X, data_Y):
    for i in range(EXAMPLES):
        x = np.array((data_X[i], 1.0))
        print("x:{}, y_original:{}, y_predict:{}".format(data_X[i], data_Y[i], h_theta(theta, x)))

def graph_loss(data):
    plt.plot(data)
    y = np.arange(len(data))
    plt.title("Loss function (SGD) vs Iteration")
    plt.xlabel("Iteration Number")
    plt.ylabel("Loss function (SGD)")
    
    os.system('mkdir -p assets')
    plt.savefig(os.path.join(BASE_DIR, 'Q1', 'assets', 'Loss_function.jpg'))
        
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
    
    os.system('mkdir -p assets')
    plt.savefig(os.path.join(BASE_DIR, 'Q1', 'assets', 'partB_hypothesis_function.jpg'))
        
    plt.legend()
    plt.show()

def update_params(theta, learning_rate, gradient):
    return np.add(theta, np.multiply(gradient, -learning_rate))

def partA(iterations, learning_rate, stopping_threshold, theta, loss, data_x, data_y):
    print("Part A")
    print("Iterations: {}".format(iterations))
    print("Stopping Threshold: {}".format(stopping_threshold))
    print("Parameters Learned: {}".format(theta))
    
    input("\nPress Enter to move to Part B")
    partB(iterations, learning_rate, stopping_threshold, theta, loss, data_x, data_y)

def partB(iterations, learning_rate, stopping_threshold, theta, loss, data_x, data_y):
    graph_y_yhat(data_x, data_y, theta)
    input("\nPress Enter to move to Part C")
    partC(iterations, learning_rate, stopping_threshold, theta, loss, data_x, data_y)

def partC(iterations, learning_rate, stopping_threshold, theta, loss, data_x, data_y):
    pass

def batch_gradient_descent(data_x, data_y):    
    global EXAMPLES
    EXAMPLES = data_x.shape[0]
    theta = np.zeros((2)) # Initialized theta to be a zero vector
    
    count1, count2 = 0, 0 # For checking convergence (SGD)
    loss1, loss2   = 0, 0
    
    i = 0 # Iteration count
    
    loss = []
    
    while True:
        i+=1
        iteration_loss, gradient = batch_gradient(data_x, data_y, theta)
        theta = update_params(theta, learning_rate, gradient)        
        
        loss.append(iteration_loss)
        
        if count1 < K_ITERATIONS:
            count1+=1
            loss1 += iteration_loss
        elif  count2 < K_ITERATIONS:
            count2+=1
            loss2 += iteration_loss
            
        if(count2 == K_ITERATIONS):
            loss1 /= BATCH_SIZE
            loss2 /= BATCH_SIZE
            count1 = 0
            count2 = 0
        
            if converge(loss1,loss2):
                partA(i, learning_rate, DELTA, theta, loss, data_x, data_y)
                # print("Iterations: {}".format(i))
                # print("Parameters Learned: {}".format(theta))
                # graph_loss(loss)
                graph_y_yhat(data_x, data_y, theta)
                return

if __name__ == '__main__':
    X = normalize(data_load('linearX.csv'))
    Y = data_load('linearY.csv')
    if X.shape[0] != Y.shape[0]:
        raise IOError("Inconsistent dimensions of data")
    batch_gradient_descent(X,Y)