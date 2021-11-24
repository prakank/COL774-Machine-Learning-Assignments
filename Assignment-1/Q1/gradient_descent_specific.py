import numpy as np

def h_theta(theta, x):
    if type(x) == int or type(x) == float:
        x = np.array([x,1])
    return np.dot(np.transpose(theta), x)

def loss_function(y, theta, x):
    temp = (y - h_theta(theta, x))
    return temp*temp

def check(theta, data_X, data_Y):
    for i in range(data_X.shape[0]):
        x = np.array((data_X[i], 1.0))
        print(h_theta(theta, x), data_Y[i])
 