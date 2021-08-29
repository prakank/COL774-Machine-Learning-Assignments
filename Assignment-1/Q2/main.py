import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLING_SIZE = 1e6

def data_load(filename):
    path = os.path.join(BASE_DIR, 'data', 'q2', filename)
    data = np.genfromtxt(path, delimiter=',')
    return data

def normalize(data):
    data = (data - data.mean(axis=0))/data.std(axis=0)
    return data

def shuffle_data(data_x, data_y):
    pass

def normal_distribution(SIZE, mean, variance):
    std = math.sqrt(variance)
    distribution = np.random.normal(loc=mean, scale=std, size=SIZE)
    return distribution

def sampling(SIZE, theta, x1_distribution, x2_distribution, noise_distribution): # theta will be [theta2 theta1 theta0] (numpy array)
    x0    = [1 for i in range(SIZE)]
    x1    = normal_distribution(SIZE, x1_distribution[0]   , x1_distribution[1])
    x2    = normal_distribution(SIZE, x2_distribution[0]   , x2_distribution[1])
    noise = normal_distribution(SIZE, noise_distribution[0], noise_distribution[1])
    x = np.stack((x2, x1, x0), axis=0)
    y = np.dot(theta.T, x) + noise
    return x,y

def h_theta(theta, x):
    if type(x) == int or type(x) == float:
        x = np.array([x,1])
    return np.dot(np.transpose(theta), x)

def loss_function(y, theta, x):
    temp = (y - h_theta(theta, x))
    return temp*temp

def update_params(theta, learning_rate, gradient):
    return np.add(theta, np.multiply(gradient, -learning_rate))

def converge(delta, loss1, loss2):
    return abs(loss1 - loss2) <= delta

def batch_gradient(theta, batch_number, batch_size, data_x, data_y): # -(1/batch_size) * summation_1_batch_size ( ( y(i) - h_theta(x_i) ) * x(i) )
    gradient = np.zeros((3))
    loss = 0
    
    for i in range(batch_size):
        index = batch_number*batch_size + i   
        x = data_x[[index],:]
        x = x.reshape(x.shape[1])
        partial_gradient = np.multiply( x,(data_y[index] - h_theta(theta, x)) )
        partial_loss     = loss_function(data_y[index], theta, x)
        
        gradient = np.add(gradient,partial_gradient)
        loss     = partial_loss + loss
        
    gradient = np.multiply(gradient,-1/batch_size)
    loss = (loss)/(2*batch_size)
    
    return loss, gradient


def stochastic_gradient_descent(learning_rate, delta, converge_iterations, batch_size, data_x, data_y):
    EXAMPLES = data_x.shape[0]
    theta = np.zeros((3)) # Initialized theta to be a zero vector [theta2 theta1 theta0]
    
    count1, count2 = 0, 0 # For checking convergence (SGD)
    loss1, loss2   = 0, 0
    
    iterations = 0 # Iteration count
    
    theta_list = []
    total_batch = EXAMPLES//batch_size
            
    while True:
        iterations+=1
        theta_list.append(theta)
        batch_number = int((iterations-1)%total_batch)
        
        iteration_loss, gradient = batch_gradient(theta, batch_number, batch_size, data_x, data_y)
        theta = update_params(theta, learning_rate, gradient)
        
        if count1 < converge_iterations:
            count1+=1
            loss1 += iteration_loss
        elif  count2 < converge_iterations:
            count2+=1
            loss2 += iteration_loss
            
        if(count2 == converge_iterations):
            loss1 /= batch_size
            loss2 /= batch_size
        
            if converge(delta, loss1, loss2):
                return (iterations, theta, theta_list)
            else:
                print(loss1, loss2)
                count1, count2 = 0, 0
                loss1, loss2   = 0, 0
                
                        
def main():
    # Data Generation
    theta = np.array([2,1,3])
    data_x, data_y = sampling(int(SAMPLING_SIZE), theta, [3,4], [-1,4], [0,2])
    
    # Data Normalization
    data_x = data_x.T
    # data_x = normalize(data_x).T
    # data_x, data_y = shuffle_data(data_x, data_y)
    
    # Trying to obtain same parameters as which were used to generate the data
    learning_rate = 0.001
    
    batch_description = [
        {
            "learning_rate": learning_rate,
            "batch_size": 100,
            "delta": 1e-4,
            "converge_iterations": 10,
        },
        {
            "learning_rate": learning_rate,
            "batch_size": 100,
            "delta": 1e-4,
            "converge_iterations": 100,
        },
        {
            "learning_rate": learning_rate,
            "batch_size": 10000,
            "delta": 1e-4,
            "converge_iterations": 10,
        },
        {
            "learning_rate": learning_rate,
            "batch_size": 1000000,
            "delta": 1e-4,
            "converge_iterations": 1,
        }
    ]
    for batch in batch_description:
        iterations, theta, theta_list = stochastic_gradient_descent(batch["learning_rate"], batch["delta"], batch["converge_iterations"], batch["batch_size"], data_x, data_y)
        print(iterations, theta)

# {
#     "learning_rate": learning_rate,
#     "batch_size": 100,
#     "delta": 1e-4,
#     "converge_iterations": 100,
# },


if __name__ == '__main__':
    main()
