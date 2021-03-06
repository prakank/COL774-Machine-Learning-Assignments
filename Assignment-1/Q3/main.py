import numpy as np
import matplotlib.pyplot as plt
import math
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Iterations = 100
MAX_ITERATIONS = 100000
MINIMUM_LOSS = 0.1
Delta = 1e-10

def data_load(filename):
    path = os.path.join(BASE_DIR, 'data', 'q3', filename)
    data = np.genfromtxt(path, delimiter=',')
    return data

def normalize(data):
    data = (data - data.mean(axis=0))/data.std(axis=0)
    return data

def hypothesis_function(theta, x):
    return np.dot(theta.T, x)

def individual_loss(x, y, theta):
    h_theta = hypothesis_function(theta, x)
    if y==0:
        return np.log(1-h_theta)
    else:
        return np.log(h_theta)

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-1.0*z))

def converge(l1, l2):
    if l1 > MINIMUM_LOSS or l2 > MINIMUM_LOSS:
        return False
    if(abs(l1-l2) <= Delta):
        return True
    return False

def loss_val(theta, dataX, dataY):
    examples = len(dataY)
    h_theta = sigmoid(np.dot(dataX, theta))
    loss = np.sum(dataY*np.log(h_theta) + (1-dataY)*np.log(1-h_theta))
    return loss/(2*dataY.shape[0])

def loss_grad(theta, dataX, dataY):
    h_theta = sigmoid(np.dot(dataX, theta))
    return -np.dot(dataX.T, dataY - h_theta)/(2*dataY.shape[0])

def HessianMatrix(theta, dataX, dataY):
    h_theta  = sigmoid(np.dot(dataX, theta))
    diagonal = np.diag(h_theta*(1-h_theta))
    # hessian is X*D*X.T (Where dimension of X is features*examples)
    hessian = np.dot(np.dot(dataX.T, diagonal), dataX)    
    hessian = hessian/(2*dataY.shape[0])    
    return hessian

def update_params(theta, gradient, hessian):
    update = np.dot(np.linalg.inv(hessian), gradient)
    return (theta - update)

def get_accuracy(X,Y,theta):
    Y_pred=sigmoid(np.dot(X,theta))
    Y_pred=np.where(Y_pred>0.5,1,0)
    return np.mean(Y_pred==Y)

    
def training(dataX, dataY):
    theta = np.zeros((dataX.shape[1]))
    loss_list = []
    iterations = 0
    while True:
        iterations+=1
        loss = loss_val(theta, dataX, dataY)
        loss_list.append(loss)
        gradient = loss_grad(theta, dataX, dataY)
        hessian = HessianMatrix(theta, dataX, dataY)
        theta = update_params(theta, gradient, hessian)
        if len(loss_list) > 1:
            l1 = loss_list[-1]
            l2 = loss_list[-2]
            if converge(l1, l2):
                return theta, iterations
        if iterations >= MAX_ITERATIONS:
            return theta, iterations
        
def predict(theta, dataX, dataY):
    predicted = sigmoid(np.dot(dataX, theta))
    for i in range(len(predicted)):
        print("Predicted Val:{}, Original Val:{}".format(round(predicted[i],2), dataY[i]))

def y_val(x, theta):
    return (-theta[0] - x*theta[1])/theta[2]

def graph(theta, dataX, dataY):
    x1_neg, x2_y_neg = [], []
    x1_pos, x2_y_pos = [], []
    theta0, theta1, theta2 = theta[0], theta[1], theta[2]
    for i in range(dataY.shape[0]):
        if(dataY[i] == 0):
            x1_neg.append(dataX[i][0]) #x1 in x2*theta2 + x1*theta1 + x0
            x2_y_neg.append(dataX[i][1]) #x2 in x2*theta2 + x1*theta1 + x0
        else:
            x1_pos.append(dataX[i][0])
            x2_y_pos.append(dataX[i][1])
    plt.scatter(x1_neg, x2_y_neg, c='green', label='Negative -> 0')
    plt.scatter(x1_pos, x2_y_pos, c='red', label='Positive   -> 1')
    
    min_x = -2.2
    max_x =  2.2
    
    point1 = (min_x, y_val(min_x, theta))
    point2 = (max_x, y_val(max_x, theta))
    
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], label="Decision Boundary", color='orange')
    plt.xlabel('Feature 1 (x1)')
    plt.ylabel('Feature 2 (x2)')
    plt.title("Logistic Regression (Newton's Method)")
    plt.legend(loc="upper right")
    
    os.system('mkdir -p assets/')
    plt.savefig(os.path.join(BASE_DIR, 'Q3', 'assets', 'Logistic_Regression.jpg'))
    
    plt.show()
            
def main():
    X_orig = normalize(data_load('logisticX.csv'))
    Y = data_load('logisticY.csv')
    # X_orig[:, [1, 0]] = X_orig[:, [0, 1]] #swap columns so as to have [ x2 x1 x0 ]
    X = np.c_[np.ones(Y.shape), X_orig] # [x0 x1 x2]
    theta, iterations = training(X, Y)
    print("Iterations: {}".format(iterations))
    print("Learned Parameters:\nTheta0: {}\nTheta1: {}\nTheta2: {}"
          .format(round(theta[0],4), round(theta[1],4), round(theta[2],4)))
    graph(theta, X_orig, Y)

if __name__ == '__main__':
    main()