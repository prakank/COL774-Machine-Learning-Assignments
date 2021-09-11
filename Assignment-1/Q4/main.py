import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

MAP = {'Alaska':0, 'Canada':1}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def data_load(filename):
    path = os.path.join(BASE_DIR, 'data', 'q4', filename)
    data = pd.read_csv(path, sep="\s+",header=None)
    data = np.asarray(data.values)
    return data

def normalize(data):
    data = (data - data.mean(axis=0))/data.std(axis=0)
    return data

def load_x(filename):
    X = data_load(filename)
    X = X.astype('float64')
    X = normalize(X)
    return X

def load_y(filename):
    Y = data_load(filename)
    Y = Y.reshape((Y.shape[0]))
    Y = np.asarray(list(map(lambda x: MAP[x], Y)))
    return Y

def graph_training(X,Y,title,filename):
    x1_neg, x2_y_neg = [], []
    x1_pos, x2_y_pos = [], []
    for i in range(Y.shape[0]):
        if(Y[i] == 0):
            x1_neg.append(X[i][0]) # 1st value in the list is taken to be the first feature i.e. Fresh water
            x2_y_neg.append(X[i][1]) # 2nd value in the list is taken to be the second feature i.e. Marine water
        else:
            x1_pos.append(X[i][0])
            x2_y_pos.append(X[i][1])
    plt.scatter(x1_neg, x2_y_neg, c='green', label='Alaska   -> 0')
    plt.scatter(x1_pos, x2_y_pos, c='red', label='Canada -> 1')
    plt.xlabel('Fresh Water (x1 normalized)')
    plt.ylabel('Marine Water (x2 normalized)')
    plt.title(title)
    plt.legend(loc="upper right")
    
    os.system('mkdir -p assets/')
    plt.savefig(os.path.join(BASE_DIR, 'Q4', 'assets', filename))
    
    plt.show()

def y_val(x,A,B):
    # x1*A[0][0] + x2*A[0][1] = B (x1 is taken to be x)
    return (B - x*A[0][0])/A[0][1]

def graph_gda(X,Y,A,B,title,filename):
    x1_neg, x2_y_neg = [], []
    x1_pos, x2_y_pos = [], []
    for i in range(Y.shape[0]):
        if(Y[i] == 0):
            x1_neg.append(X[i][0]) # 1st value in the list is taken to be the first feature i.e. Fresh water
            x2_y_neg.append(X[i][1]) # 2nd value in the list is taken to be the second feature i.e. Marine water
        else:
            x1_pos.append(X[i][0])
            x2_y_pos.append(X[i][1])
    plt.scatter(x1_neg, x2_y_neg, c='green', label='Alaska   -> 0')
    plt.scatter(x1_pos, x2_y_pos, c='red', label='Canada -> 1')
    
    min_x = -1
    max_x =  2.2
    
    # x1*A[0][0] + x2*A[0][1] = B
    point1 = (min_x, y_val(min_x, A, B))
    point2 = (max_x, y_val(max_x, A, B))
    
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], label="Decision Boundary", color='orange')    
    plt.xlabel('Fresh Water (x1 normalized)')
    plt.ylabel('Marine Water (x2 normalized)')
    plt.title(title)
    plt.legend(loc="upper right")
    
    os.system('mkdir -p assets/')
    plt.savefig(os.path.join(BASE_DIR, 'Q4', 'assets', filename))
    
    plt.show()

def part5(X,Y,mu0,mu1,phi,sigma,sigma0,sigma1):
    pass

def part4(X,Y,mu0,mu1,phi,sigma,sigma0,sigma1):
    print("mu0 = {}".format(mu0))
    print("mu1 = {}\n".format(mu1))
    print("Sigma0  = {}\n".format(sigma0))
    print("Sigma1  = {}\n".format(sigma1))
    
    # input("\nPress Enter to move to Part E")
    part5(X,Y,mu0,mu1,phi,sigma,sigma0,sigma1)

def part3(X,Y,mu0,mu1,phi,sigma,sigma0,sigma1):
    mu0 = mu0.reshape(mu0.shape[0],1)
    mu1 = mu1.reshape(mu1.shape[0],1)
    
    C     = np.log((1-phi)/phi)
    term2 = np.dot( (np.dot(mu1.T, np.linalg.inv(sigma))), mu1) + np.dot( (np.dot(mu0.T, np.linalg.inv(sigma))), mu0)
    term2 = term2[0][0]
    term1 = np.dot((mu1 - mu0).T, np.linalg.inv(sigma))
    
    # term1*x = term2/2 + C
    # A*x = B
    
    A = term1
    B = (term2/2) + C
    title = "Linear Separator (GDA)"
    filename = "Linear_Separator_GDA.jpg"    
    graph_gda(X,Y,A,B,title,filename)
    
    # input("\nPress Enter to move to Part D")
    part4(X,Y,mu0,mu1,phi,sigma,sigma0,sigma1)
                
def part2(X,Y,mu0,mu1,phi,sigma,sigma0,sigma1):
        
    title = "Distribution of Training Data"
    filename = "Distribution_of_Training_Data.jpg"
    graph_training(X,Y,title,filename)
    
    # input("\nPress Enter to move to Part C")
    part3(X,Y,mu0,mu1,phi,sigma,sigma0,sigma1)
    
def part1(X,Y):
    bool_filter_zero = Y == 0
    bool_filter_one = Y == 1
    data_zero = X[bool_filter_zero]
    data_one  = X[bool_filter_one]

    mu0 = data_zero.mean(axis=0).reshape((X.shape[1]))
    mu1 = data_one.mean(axis=0).reshape((X.shape[1]))
    phi = data_one.shape[0]/X.shape[0]
    
    sigma = np.zeros((X.shape[1], X.shape[1]))
    sigma0 = np.zeros((X.shape[1], X.shape[1]))
    sigma1= np.zeros((X.shape[1], X.shape[1]))
    
    for i in range(X.shape[0]):
        if(Y[i] == 0):
            val = X[i] - mu0
            val = val.reshape(val.shape[0],1)
            val = val*val.T
            sigma0 += val
            sigma+=val
        else:
            val = X[i] - mu1
            val = val.reshape(val.shape[0],1)
            val = val*val.T
            sigma1 += val        
            sigma+=val
        
    sigma  = sigma/X.shape[0]
    sigma0 = sigma0/data_zero.shape[0]
    sigma1 = sigma1/data_one.shape[0]
        
    print("mu0 = {}".format(mu0))
    print("mu1 = {}\n".format(mu1))
    print("Sigma  = {}\n".format(sigma))
    
    # input("\nPress Enter to move to Part B")
    part2(X,Y,mu0,mu1,phi,sigma,sigma0,sigma1)
        
def main():
    X = load_x("q4x.dat")
    Y = load_y("q4y.dat")
    part1(X,Y)

if __name__ == '__main__':
    main()