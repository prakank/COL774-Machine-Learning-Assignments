import sys
import csv
import time
import random
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Reading the input files into matrices. 
# Distance specifies the number of classes essentially (offset because of one hot encoding)
# Distance is the last set of columns which are to be used as output labels
def read_file(datapath,distance):
	full_data = pd.read_csv(datapath,header=None,dtype=int)
	full_data_arr = np.array(full_data)
	full_data_shape = full_data.shape
	x = np.asmatrix(full_data_arr[:,0:full_data_shape[1]-distance])
	y = np.asmatrix(full_data_arr[:,full_data_shape[1]-distance:full_data_shape[1]])
	return (x,y)

# Activation functions and their derivatives
def sigmoid_activation(a):
	# Following implementation avoids overflows
	a1 = np.multiply(a>=0,a)
	a2 = np.multiply(a<0,a)
	return np.add(1/(1+np.exp(-a1)),np.divide(np.exp(a2),(1+np.exp(a2)))) - 0.5
	
def sigmoid_derivative(a):
	return np.multiply(sigmoid_activation(a),1-sigmoid_activation(a))

def relu_activation(a):
	return np.multiply(a>0,a)

def relu_derivative(a):
	return np.multiply(a>0,np.ones(a.shape,dtype=float))

def leaky_relu_activation(a):
	return np.add(np.multiply(a>0,a),np.multiply(0.01*a,a<=0))
	
def leaky_relu_derivative(a):
	return np.add(np.multiply(1,a>0),np.multiply(0.01,a<=0))

# Normalization function
def normalization(mat):
	mean = np.mean(mat,axis=0)
	var = np.var(mat,axis=0)
	# To avoid division by zero add 1 where it's 0
	var = var + np.multiply(1,var==0)
	return np.divide(np.subtract(mat,mean),var)

# Depending on architecture specified initializes weights and biases of all layers except the input layer
# Have used xavier initialization
def initialize_params(arch_details):
	np.random.seed(1)
	params = {}
	for l in range(1,len(arch_details)):
		params["W"+str(l)] = np.random.normal(0,1,(arch_details[l],arch_details[l-1]))*np.sqrt(2.0/arch_details[l-1])
		params["b"+str(l)] = np.zeros((arch_details[l],1),dtype=float)
	return params

# Forward propagation
# Returns a dictionary of various activations happening in hidden layers
def forward_prop(params,data_x,activation):
	forward_pass = {}
	num_layers = (int)(len(params)/2)
	x = np.transpose(data_x)
	forward_pass["a0"] = x

	if activation=="logistic":
		for i in range(num_layers-1):
			x = np.dot(params["W"+str(i+1)],x) + params["b"+str(i+1)]
			forward_pass["z"+str(i+1)] = x
			x = sigmoid_activation(x)
			forward_pass["a"+str(i+1)] = x
	
	elif activation=="lr":
		for i in range(num_layers-1):
			x = np.dot(params["W"+str(i+1)],x) + params["b"+str(i+1)]
			forward_pass["z"+str(i+1)] = x
			x = leaky_relu_activation(x)
			forward_pass["a"+str(i+1)] = x
	
	else:
		for i in range(num_layers-1):
			x = np.dot(params["W"+str(i+1)],x) + params["b"+str(i+1)]
			forward_pass["z"+str(i+1)] = x
			x = relu_activation(x)
			forward_pass["a"+str(i+1)] = x

	x = np.dot(params["W"+str(num_layers)],x) + params["b"+str(num_layers)]
	forward_pass["z"+str(num_layers)] = x
	x = sigmoid_activation(x)
	forward_pass["a"+str(num_layers)] = x
	return forward_pass

# Cross entropy loss
def loss_function(output_layer_output,actual_output):
	loss0 = np.multiply(actual_output,np.log(np.add(output_layer_output,np.multiply(1,output_layer_output==0))))
	loss1_0 = np.multiply(1-actual_output,np.log(np.add(1-output_layer_output,np.multiply(1,output_layer_output==1))))
	loss = -1*np.add(loss0,loss1_0)
	mean_loss_over_batch = np.mean(loss,axis=1)
	return np.sqrt(np.dot(np.transpose(mean_loss_over_batch),mean_loss_over_batch)[0,0])

# Back propagation function
def backward_prop(params,forward_pass,learning_rate,y_data,activation):
	der_dict = {}
	new_params = {}
	m = y_data.shape[1]

	# derivative for last layer of network
	der_output = forward_pass["a"+str((int)(len(params)/2))] - y_data
	der_dict["dZ"+str((int)(len(params)/2))] = der_output
	
	if activation=="logistic":
		for i in range((int)(len(params)/2) - 1,0,-1):
			der_output = np.multiply(np.dot(np.transpose(params["W"+str(i+1)]),der_dict["dZ"+str(i+1)]),sigmoid_derivative(forward_pass["z"+str(i)]))
			der_dict["dZ"+str(i)] = der_output

		for i in range(1,(int)(len(params)/2) +1):
			new_params["W"+str(i)] = params["W"+str(i)] - (learning_rate/m)*np.dot(der_dict["dZ"+str(i)],np.transpose(forward_pass["a"+str(i-1)]))
			new_params["b"+str(i)] = params["b"+str(i)] - (learning_rate/m)*np.sum(der_dict["dZ"+str(i)],axis=1)
	
	elif activation=="lr":
		for i in range((int)(len(params)/2) - 1,0,-1):
			der_output = np.multiply(np.dot(np.transpose(params["W"+str(i+1)]),der_dict["dZ"+str(i+1)]),leaky_relu_derivative(forward_pass["z"+str(i)]))
			der_dict["dZ"+str(i)] = der_output

		for i in range(1,(int)(len(params)/2) +1):
			new_params["W"+str(i)] = params["W"+str(i)] - (learning_rate/m)*np.dot(der_dict["dZ"+str(i)],np.transpose(forward_pass["a"+str(i-1)]))
			new_params["b"+str(i)] = params["b"+str(i)] - (learning_rate/m)*np.sum(der_dict["dZ"+str(i)],axis=1)

	else:
		for i in range((int)(len(params)/2) - 1,0,-1):
			der_output = np.multiply(np.dot(np.transpose(params["W"+str(i+1)]),der_dict["dZ"+str(i+1)]),relu_derivative(forward_pass["z"+str(i)]))
			der_dict["dZ"+str(i)] = der_output

		for i in range(1,(int)(len(params)/2) +1):
			new_params["W"+str(i)] = params["W"+str(i)] - (learning_rate/m)*np.dot(der_dict["dZ"+str(i)],np.transpose(forward_pass["a"+str(i-1)]))
			new_params["b"+str(i)] = params["b"+str(i)] - (learning_rate/m)*np.sum(der_dict["dZ"+str(i)],axis=1)
	return new_params

# Prediction on data_x point using params
def prediction(params,data_x,activation):
	forward_pass = {}
	x = np.transpose(data_x)
	for i in range((int)(len(params)/2)):
		x = np.dot(params["W"+str(i+1)],x) + params["b"+str(i+1)]
		forward_pass["z"+str(i+1)] = x
		x = sigmoid_activation(x)
		forward_pass["a"+str(i+1)] = x
	output = np.exp(forward_pass["a"+str((int)(len(params)/2))])
	summer = np.sum(output,axis=0)
	output = np.divide(output,summer)
	return np.argmax(output,axis=0)

# Plotting confusion matrix
def draw_confusion_matrix(confatrix):
	plt.imshow(confatrix)
	plt.title("Confusion Matrix")
	plt.colorbar()
	plt.set_cmap("Greens")
	plt.ylabel("True labels")
	plt.xlabel("Predicted label")
	plt.show()

# MAIN FUNCTION
def main():
	config_datapath = sys.argv[1]
	train_datapath = sys.argv[2]
	test_datapath = sys.argv[3]

	config = open(config_datapath,'r')
	num_input = (int)(config.readline())
	num_outputs = (int)(config.readline())
	batch_size = (int)(config.readline())
	num_hidden_layers = (int)(config.readline())
	hidden_layers = [(int)(x) for x in config.readline()[:-1].split(' ')]
	activation = (config.readline()[:-1])
	learning_rate_variation = config.readline()
	config.close()
	
	if activation == "sigmoid": 
		activation = "logistic"
		epochs = 1000
	else:
		epochs = 3000

	(train_x,train_y) = read_file(train_datapath,num_outputs)
	(test_x,test_y) = read_file(test_datapath,num_outputs)

	counter = 0
	epsilon = 0.000001
	tolerance = 0.0001
	cost_list = []
	learning_rate = 0.1
	learning_rate_limit = 1e-10
	error = 10
	num_datapoints = len(train_x)
	num_batches = (int)(num_datapoints/batch_size)
	neurons_list = [num_input]
	neurons_list.extend(hidden_layers)
	neurons_list.append(num_outputs)
	params = initialize_params(neurons_list)
	
	time1 = time.clock()
	
	while error>epsilon and counter<epochs and learning_rate>learning_rate_limit:
		error = 0
		for batch_counter in range(num_batches):
			begin = batch_counter*batch_size
			end = begin+batch_size
			if batch_counter==num_batches-1:
				end = num_datapoints
			forward_pass  = forward_prop(params,train_x[begin:end,:],activation)
			loss_mag = loss_function(forward_pass["a"+str((int)(len(params)/2))],np.transpose(train_y[begin:end,:]))
			params = backward_prop(params,forward_pass,learning_rate,np.transpose(train_y[begin:end,:]),activation)
			error+=((1.0*loss_mag)/(end-begin+1))
		print(str(counter) + " : " + str(error))
		cost_list.append(error)
		if learning_rate_variation=="variable" and len(cost_list)>=3:
			if cost_list[len(cost_list)-2] - cost_list[len(cost_list)-1]<tolerance and cost_list[len(cost_list)-3] - cost_list[len(cost_list)-2]<tolerance:
				learning_rate = learning_rate/5
		counter+=1
	
	time2 = time.clock()
	print(str(time2-time1) + " taken for training")
	
	print("PREDICTION ON TRAINING DATA")
	predicted_train = np.transpose(prediction(params,train_x,activation))
	confatrix_train = confusion_matrix(np.argmax(train_y,axis=1),predicted_train)
	print(confatrix_train)
	draw_confusion_matrix(confatrix_train)
	print(accuracy_score(np.argmax(train_y,axis=1),predicted_train))

	print("PREDICTION ON TESTING DATA")
	predicted_test = np.transpose(prediction(params,test_x,activation))
	confatrix_test = confusion_matrix(np.argmax(test_y,axis=1),predicted_test)
	print(confatrix_test)
	# draw_confusion_matrix(confatrix_test)
	print(accuracy_score(np.argmax(test_y,axis=1),predicted_test))

	# plt.plot(cost_list)
	# plt.show()
	
if __name__ == "__main__":
	main()