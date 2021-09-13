from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter 
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLING_SIZE = 1e6
NOISE = True
MINIMUM_LOSS = 1

def data_load(filename):
    path = os.path.join(BASE_DIR, 'data', 'q2', filename)
    data = np.genfromtxt(path, delimiter=',',skip_header=1)
    return data

def shuffle_data(data_x, data_y):
    data_y = data_y.reshape(data_y.shape[0],1)
    data = np.concatenate((data_x, data_y), axis=1)
    np.random.shuffle(data)
    data_x = data[:,[0,1,2]]
    data_y = data[:,3]
    return data_x,data_y

def normal_distribution(SIZE, mean, variance):
    std = math.sqrt(variance)
    distribution = np.random.normal(loc=mean, scale=std, size=SIZE)
    return distribution

def sampling(SIZE, theta, x1_distribution, x2_distribution, noise_distribution): # theta will be [theta2 theta1 theta0] (numpy array)
    x0    = np.ones(SIZE, dtype=float)
    x1    = normal_distribution(SIZE, x1_distribution[0]   , x1_distribution[1])
    x2    = normal_distribution(SIZE, x2_distribution[0]   , x2_distribution[1])
    noise = normal_distribution(SIZE, noise_distribution[0], noise_distribution[1])
    x = np.stack((x2, x1, x0), axis=0)
    y = np.dot(theta.T, x)
    if NOISE:
        y += noise
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
    if loss1 >= MINIMUM_LOSS or loss2>=MINIMUM_LOSS:
        return False
    return abs(loss1 - loss2) <= delta

def batch_gradient(theta, batch_number, batch_size, data_x, data_y): # -(1/batch_size) * summation_1_batch_size ( ( y(i) - h_theta(x_i) ) * x(i) )
    start_x = batch_number*batch_size
    end_x   = min(batch_number*batch_size + batch_size-1,data_x.shape[0]-1)
    rows    = np.linspace(start_x,end_x,end_x-start_x+1,dtype='int64')
    x_ = data_x[rows, :]
    y_ = data_y[rows]
    x = x_.copy()
    y = y_.copy()        
    
    h = np.asarray(list(map(lambda val: h_theta(theta,val), x)))
    y = y - h
    y = np.asarray([y]*data_x.shape[1]).T
    
    gradient = np.multiply((np.multiply(x,y).sum(axis=0)),-1/batch_size)
    
    y_loss = y_.copy()
    y_loss = np.asarray(list(map(lambda val1,val2: loss_function(val1, theta, val2), y_loss, x)))
    loss = (y_loss.sum())/(2*batch_size)
    return loss, gradient
    
    # print(loss, gradient)
                
    # # Optimize this loop
    # for i in range(batch_size):
    #     index = batch_number*batch_size + i
    #     x = data_x[[index],:]
    #     x = x.reshape(x.shape[1])
    #     partial_gradient = np.multiply( x,(data_y[index] - h_theta(theta, x)) )
    #     partial_loss     = loss_function(data_y[index], theta, x)
        
    #     gradient = np.add(gradient,partial_gradient)
    #     loss     = partial_loss + loss
        
    # gradient = np.multiply(gradient,-1/batch_size)
    # loss = (loss)/(2*batch_size)
    
    # return loss, gradient

def stochastic_gradient_descent(learning_rate, delta, converge_iterations, batch_size, ITERATION_LIMIT,data_x, data_y):
    EXAMPLES = data_x.shape[0]
    theta = np.zeros((3)) # Initialized theta to be a zero vector [theta2 theta1 theta0]
    
    count1, count2 = 0, 0 # For checking convergence (SGD)
    loss1, loss2   = 0, 0    
    iterations = 0 # Iteration count
        
    theta_list = []
    total_batch = EXAMPLES//batch_size
            
    while True:
        iterations+=1
        theta_list.append(np.flip(theta))
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
            loss1 /= converge_iterations
            loss2 /= converge_iterations
        
            if converge(delta, loss1, loss2):
                print("\nLoss 1: {}, Loss 2:{}".format(loss1, loss2))
                theta_list = np.asarray(theta_list)
                return (iterations, theta, theta_list)
            else:
                print("Iteration Number: {}, Loss 1: {}, Loss 2:{}".format(int(iterations),round(loss1,11), round(loss2,11)))
                count1, count2 = 0, 0
                loss1, loss2   = 0, 0
        if(iterations >=  ITERATION_LIMIT):
            theta_list = np.asarray(theta_list)
            return (iterations, theta, theta_list)

def predict(theta, data_x, data_y):
    count = 0
    for i in range(data_x.shape[0]):
        x = data_x[i]
        predicted = h_theta(theta, x)
        orig = data_y[i]        
        if abs(orig - predicted) > 1:
            count+=1
            print("x:", x)
            print("Original:{}, Predicted:{}".format(orig, predicted))
    print("Total: " + str(count) + "\n")

def error_calc(theta, data_x,data_y):
    error_ = np.dot(data_x,theta).reshape(data_y.shape[0],1)
    error_ = error_ - data_y
    error_ = ((np.multiply(error_, error_)).sum())/(2*data_y.shape[0])
    return error_

def error(theta, theta_orig, data):
    # theta = [theta0, theta1, theta2]
    data_x = data[:,[0,1]]
    data_y = data[:,[2]]    
    data_x = np.c_[np.ones(data_y.shape[0]), data_x]
    
    error_pred = error_calc(theta, data_x, data_y)
    error_orig = error_calc(theta_orig, data_x, data_y)
    
    return error_pred, error_orig

def graph(theta_list,frames_size,title,filename):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Theta0', fontsize=14, labelpad=10)
    ax.set_ylabel('Theta1', fontsize=14, labelpad=10)
    ax.set_zlabel('Theta2', fontsize=14, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

    graph = ax.scatter([], [], [], marker='o', c='blue', s=20)
    gap,opacity = 0.3, 1
    
    ax.set_zlim(np.amin(theta_list[:,2])-gap, np.amax(theta_list[:,2])+gap)
    ax.set_ylim(np.amin(theta_list[:,1])-gap, np.amax(theta_list[:,1])+gap)
    ax.set_xlim(np.amin(theta_list[:,0])-gap, np.amax(theta_list[:,0])+gap)
    
    graph.set_alpha(opacity)
    x_data, y_data, z_data = [], [], []

    def animation(i):
        z_data.append(theta_list[i,2])
        y_data.append(theta_list[i,1])
        x_data.append(theta_list[i,0])                
        graph._offsets3d = (x_data, y_data, z_data)
        return graph

    anim = FuncAnimation(fig, animation, frames=np.arange(0, theta_list.shape[0], frames_size), interval=10, repeat_delay=3000, blit=False)    

    plt.show()
    os.system('mkdir -p assets')
    anim.save(os.path.join(BASE_DIR, 'Q2', 'assets', filename), writer='Pillow')


def main():
    # Data Generation
    theta_orig = np.array([2,1,3])
    data_x, data_y = sampling(int(SAMPLING_SIZE), theta_orig, [3,4], [-1,4], [0,2])
    # Noise is literally messing up the data
    # Min and Max values in noise with variance 2 is: -7.46, 7.06   :)    
    
    # Data Shuffling
    data_x = data_x.T
    data_x, data_y = shuffle_data(data_x, data_y)
    
    # Trying to obtain same parameters as which were used to generate the data
    learning_rate = 0.001
    
    batch_description = [
        {
            "learning_rate": learning_rate,
            "batch_size": 1,
            "delta": 1e-10,
            "frames":1000,
            "converge_iterations": 500,
            "ITERATION_LIMIT": 100000
        },
        {
            "learning_rate": learning_rate,
            "batch_size": 100,
            "delta": 1e-9,
            "frames":500,
            "converge_iterations": 500,
            "ITERATION_LIMIT": 50000
        },
        {
            "learning_rate": learning_rate,
            "batch_size": 10000,
            "delta": 1e-3,
            "frames":10,
            "converge_iterations": 5,
            "ITERATION_LIMIT": 1000
        },
        {
            "learning_rate": learning_rate,
            "batch_size": 1000000,
            "delta": 1e-1,
            "frames":1,
            "converge_iterations": 1,
            "ITERATION_LIMIT": 50
        }
    ]
    
    # batch_description = [
    #     {
    #         "learning_rate": learning_rate,
    #         "batch_size": 1,
    #         "delta": 1e-6,
    #         "frames":100,
    #         "converge_iterations": 1000,
    #         "ITERATION_LIMIT": 10000
    #     }
    # ]
    
    data_test = data_load('q2test.csv')
    
    start = time.time()
    i = len(batch_description)
    
    for batch in batch_description:
        print("Batch Details:\nLearning Rate:{}, Batch Size: {}\nDelta: {}, Converge Iterations: {}\nMax Allowed Iterations: {}\n"
              .format(batch["learning_rate"], batch["batch_size"], batch["delta"], 2*batch["converge_iterations"], batch["ITERATION_LIMIT"]))
        
        iterations, theta, theta_list = stochastic_gradient_descent(batch["learning_rate"], batch["delta"], batch["converge_iterations"], batch["batch_size"], batch['ITERATION_LIMIT'],data_x, data_y)        
        i-=1
        
        print("\nIterations:{}, Batch Size: {}, Delta: {}, Theta: {}".format(iterations,batch["batch_size"],batch["delta"],np.flip(theta)))
        
        error_pred, error_orig = error(np.flip(theta),np.flip(theta_orig),data_test)
        
        print("\nError with original theta: {}\nError with  learned theta: {}".format(round(error_orig,5), round(error_pred,5)))
        print("Difference in the test error or original and learned theta: {}".format(abs(round(error_orig-error_pred,5))))
        end = time.time()
        print("Time: {} sec\n\n\n".format(round(end-start,2)))
        
        title    = "Theta Movement (Batch Size:" + str(batch["batch_size"]) + ")"
        filename = "Theta_Movement_r_" + str(batch["batch_size"]) + ".gif"
        
        graph(theta_list,batch["frames"],title,filename)
                
        # if i>0:
        #     input("\nPress Enter for Next Batch Size\n")
            
        start = time.time()

if __name__ == '__main__':
    main()
