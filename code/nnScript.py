import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from math import pow
from math import exp
import pickle
import time


def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    sigmoid = 1.0 / (1.0 + np.exp(-1.0 * z))
    return sigmoid #your code here
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    #mat = loadmat('C:/Users/Karthik/Desktop/Spring 2015/Machine Learning/Projects/project-final/basecode/mnist_all.mat')
    mat = loadmat('mnist_all.mat')


    #loads the MAT object as a Dictionary
    
    #Pick a reasonable size for validation data

    #Your code here
    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])
    
    f = 255.0
    label_no = 10
    train_size = 50000
    val_size = 10000
    test_size = 10000
    image_size=784
    
    #Extracting the training data, validation data and test data
    trainy= np.empty((1,image_size))
    train_l=np.empty((1,1))
    testy = np.empty((1,image_size))
    test_l=np.empty((1,1))
         
    for i in range(label_no):
        trainx = mat.get('train'+str(i))
        l_train=trainx[:,:1]
        l_train.fill(i)
        trainy = np.concatenate((trainy,trainx),axis=0)
        train_l = np.concatenate((train_l,l_train),axis=0)
      
    trainy = trainy[1:, :]
    trainy = np.double(trainy)
    trainy = trainy/f
    train_l=train_l[1:,:]
   
    for i in range(label_no):
        testx = mat.get('test'+str(i))
        l_test=testx[:,:1]
        l_test.fill(i)
        testy = np.concatenate((testy,testx),axis=0)
        test_l=np.concatenate((test_l,l_test),axis=0)
      
    testy = testy[1:, :]    
    testy = np.double(testy)
    testy = testy/f
    test_l=test_l[1:,:]
    
    #feature selection to increase the efficiency or run time
    all_data = np.concatenate((trainy,testy),axis=0)
    
    N = np.all(all_data == all_data[0,:], axis = 0)
    all_data = all_data[:,~N]

    trainy = all_data[0:(train_size+val_size),:]
    testy = all_data[(train_size+val_size):,:]
    
    #shuffling the data
    ordr = range(trainy.shape[0])
    rand_ord = np.random.permutation(ordr)
    trainy1 = trainy[rand_ord[0:train_size],:]
    trainy2 = trainy[rand_ord[train_size:],:]
    train_l1 = train_l[rand_ord[0:train_size],:]
    train_l2 = train_l[rand_ord[train_size:],:]

    train_data = trainy1
    validation_data = trainy2
    test_data = testy
    train_label = train_l1
    validation_label = train_l2
    test_label = test_l
        
    return train_data, train_label, validation_data, validation_label, test_data, test_label    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    train_size = training_data.shape[0]
    
    #feed forward mechanism
    labels = np.array([])
    
    w1t=np.transpose(w1)  
    bias1 = np.ones((train_size,1))
    data=np.concatenate((training_data,bias1),axis=1)
    net1 = data.dot(w1t)
    sigma1 =sigmoid(net1)
    w2t = np.transpose(w2)
    bias2 = np.ones((train_size,1))
    sigma1=np.concatenate((sigma1,bias2),axis=1)
    net2 = sigma1.dot(w2t)
    sigma2=sigmoid(net2)
    oil=sigma2
    yil = training_label
    p = np.ones((train_size,n_class))
    
    #Backward propagation
    temp = np.zeros((train_size,n_class))
    
    for i in range(yil.shape[0]):
        t = yil[i,0]
        temp[i,t] = 1
    
    yil = temp
    pq = p-yil
    qr = p-oil
    ln_oil = np.log(oil)
    ln_qr = np.log(qr)
    
    #Calculating the error function J    
    J = np.zeros((train_size,10))
    
    J = (yil*ln_oil)+((pq*ln_qr))
    
    #obj_val_temp = -(obj_val_temp/train_size)
    J = -(J/train_size)
    
    #calculating the error function after regularization
    
    d1=w1**2
    d2=w2**2
    d1=d1.sum()
    d2=d2.sum()
    d3=d1+d2  
   
    J=J.sum()+(lambdaval/(2*train_size))*(d3)
        
    obj_val = J
    
    #Calculating the gradient for output-node weights
    
    grad_w2 = np.zeros((n_class,n_hidden+1))
    out_diff = oil-yil
    grad_w2 = (np.transpose(out_diff)).dot(sigma1)
    grad_w2 = grad_w2 + (lambdaval*w2)
    grad_w2 = grad_w2/train_size
 
    #Calculating the gradient for hidden-node weights
    grad_w1 = np.zeros((n_hidden,n_input+1))
    
    sigma_temp = np.ones((train_size,n_hidden+1))
    sigma_temp = sigma_temp - sigma1
        
    term1 = np.multiply(sigma_temp,sigma1)
    term2 = np.dot(out_diff,w2)
    term3 = np.multiply(term1,term2)
    
    grad_w1 = np.dot(np.transpose(term3),data) 
    
    #since gradient is not calculated for the bias node in the hidden layer we make the corresponding vector in grad_w2 as zeros.
    grad_w1 = np.delete(grad_w1,n_hidden,0)
    
    grad_w1 = grad_w1 + (lambdaval*w1)    
    grad_w1 = grad_w1/train_size
        
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
        
    print obj_val
    
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
       
    return (obj_val,obj_grad)


def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    predict_size = data.shape[0]
    
    #predicting the output values
    labels = np.array([])
    w1t=np.transpose(w1)  
    bias1 = np.empty((predict_size,1))
    bias1.fill(1.0)
    data=np.concatenate((data,bias1),axis=1)
    net1 = data.dot(w1t)
    sigma1 =sigmoid(net1)
    w2t = np.transpose(w2)
    bias2=np.empty((predict_size,1))
    bias2.fill(1.0)
    sigma1=np.concatenate((sigma1,bias2),axis=1)
    net2=sigma1.dot(w2t)
    sigma2=sigmoid(net2)

    #Converting the outputs obtained in the form of labels of size 50000X1
    labels_temp = np.zeros((predict_size,1))
    for i in range(sigma2.shape[0]):
        max_index = np.argmax(sigma2[i],axis=0)
        labels_temp[i,0] = max_index
    
    labels = labels_temp    
    return labels
    



"""**************Neural Network Script Starts here********************************"""

start_time = time.time()

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 16;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.2;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient.  Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

end_time = time.time()

#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

Learning_time = end_time - start_time
print "Learning time in secs is " 
print Learning_time 

#Writing the parameters on to the pickle file

#weights = { "W1":w1 , "W2": w2 , "optimal_n_hidden":n_hidden , "optimal_lambda":0.2 }
#pickle.dump( weights, open( "params_final.pickle", "wb" ) )
#parameters = pickle.load( open( "params_final.pickle", "rb" ) )
#print parameters