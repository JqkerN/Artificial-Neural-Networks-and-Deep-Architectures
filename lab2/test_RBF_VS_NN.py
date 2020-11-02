from rbf import RBF
from multi_layer_2 import neuralNetwork
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
import matplotlib.pylab as pylab
import random


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (5, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)



def main():

    # RBF
    mu = np.arange(0,2*math.pi, 0.2)
    dim=mu.shape[0]
    rbf = RBF(dim)
    weights         = rbf.initWeights()

    # NN
    nodes = mu.shape[0]
    bias = False
    NN = neuralNetwork(bias=bias, layers=[nodes, 1])
    NN.initWeights()


    ## Dataset for RBF
    x_train = np.arange(0,2*math.pi,0.1)
    x_train = np.random.permutation(x_train)
    split = 0.1
    nr_datapoints = x_train.shape[0]
    
    x_train = x_train[round(split*nr_datapoints):]
    x_valid =  x_train[0:round(split*nr_datapoints)]
    x_test = np.arange(0.05,2*math.pi,0.1)
    
    # SINUS
    # y_train,_ = rbf.generateData(x_train, noise=True)
    # y_valid,_ = rbf.generateData(x_valid, noise=True)
    # y_test,_ = rbf.generateData(x_test, noise=True)
    # SQUARE
    _,y_train = rbf.generateData(x_train, noise=True)
    _,y_valid = rbf.generateData(x_valid, noise=True)
    _,y_test = rbf.generateData(x_test, noise=True)


    ## Dataset for Neural Network
    x_train_NN = x_train.reshape((1, x_train.shape[0]))
    y_train_NN = y_train.reshape((1, y_train.shape[0]))

    x_valid_NN = x_valid.reshape((1, x_valid.shape[0]))
    y_valid_NN = y_valid.reshape((1, y_valid.shape[0]))

    x_test_NN = x_test.reshape((1, x_test.shape[0]))
    y_test_NN = y_test.reshape((1, y_test.shape[0]))




    ## Training the NN
    epoch_vec, loss_vec_train, loss_vec_val = NN.train(x_train=x_train_NN, y_train=y_train_NN, x_valid=x_valid_NN, y_valid=y_valid_NN, epochs=2000000, eta=0.001, alpha=0)
    print("Patient=1 triggerad at epoch: ", epoch_vec[-1])
    plt.figure("Learning Curve")
    plt.plot(epoch_vec, loss_vec_train)
    plt.plot(epoch_vec, loss_vec_val)
    plt.legend(("Training loss", "Validation loss"))
    




    
    ## Training the RBF
    sigma = 0.5
    weights, error  = rbf.train_LS(x_train, y_train, weights, mu, sigma)
    
    ## Evaluation 
    y_test_RBF = rbf.evaluation_LS(x_test, weights, mu, sigma)
    residual_error = np.sum(abs(y_test - y_test))/y_test.shape[0]


    output = NN.classify(x_test_NN)
    plt.figure("Results")
    plt.title('Single Hidden Layer VS RBF')
    plt.plot(x_test, y_test_RBF, label='RBF')
    plt.plot(x_test_NN[0], output[0], label='Single Hidden Layer')
    plt.plot(x_test, y_test, label='True Value', color='black')    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    plt.show()
if __name__ == "__main__":
    main()