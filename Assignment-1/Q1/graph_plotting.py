import numpy as np
import matplotlib.pyplot as plt
import math
from gradient_descent_specific import *

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
 