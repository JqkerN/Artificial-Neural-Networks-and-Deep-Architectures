from rbf import RBF
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (5, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# Add zero-mean gaussian noize with variance 0.1

def threshold(dataset,threshold=0):
    thresholded = [1 if datapoint > threshold else -1 for datapoint in dataset]
    thresholded=np.array(thresholded)
    return thresholded

def sinus_LS(x_test, x_train, mu, sigma):
    dim=mu.shape[0]
    rbf = RBF(dim)
    sinus, _   = rbf.generateData(x_train, noise=True)
    sinus_test, _ = rbf.generateData(x_test, noise=True)

    ## Init and train.
    weights         = rbf.initWeights()
    weights, error  = rbf.train_LS(x_train, sinus, weights, mu, sigma)
    
    ## Evaluation 
    y_test = rbf.evaluation_LS(x_test, weights, mu, sigma)
    residual_error = np.sum(abs(y_test - sinus_test))/y_test.shape[0]
    return residual_error, y_test, sinus_test

def square_LS(x_test, x_train, mu, sigma):
    dim=mu.shape[0]
    rbf = RBF(dim)
    _, square   = rbf.generateData(x_train, noise=True)
    _, square_test = rbf.generateData(x_test, noise=True)

    ## Init and train.
    weights         = rbf.initWeights()
    weights, error  = rbf.train_LS(x_train, square, weights, mu, sigma)
    
    ## Evaluation 
    y_test = rbf.evaluation_LS(x_test, weights, mu, sigma)
    residual_error = np.sum(abs(y_test - square_test))/y_test.shape[0]
    return residual_error, y_test, square_test

def sinus_delta(x_test, x_train, mu, sigma):
    dim=mu.shape[0]
    rbf = RBF(dim)
    
    ## Generate data
    sinus, _   = rbf.generateData(x_train, noise=True)
    sinus_test, _ = rbf.generateData(x_test, noise=True)
    sinus_test = sinus_test.reshape((sinus_test.shape[0],1))
    ## Init and train.
    weights         = rbf.initWeights()
    weights, _, _  = rbf.train_DELTA(x_train, weights, mu, sigma)

    ## Evaluation 
    y_test = rbf.evaluation_DELTA(x_test, weights, mu, sigma)
    tmp = abs(y_test - sinus_test)
    residual_error = np.sum(abs(y_test - sinus_test))/y_test.shape[0]
    return residual_error, y_test, sinus_test

def square_delta(x_test, x_train, mu, sigma):
    dim=mu.shape[0]
    rbf = RBF(dim)
    _, square   = rbf.generateData(x_train, noise=True)
    _, square_test = rbf.generateData(x_test, noise=True)
    square_test = square_test.reshape((square_test.shape[0],1))

    ## Init and train.
    weights         = rbf.initWeights()
    weights, _, _  = rbf.train_DELTA(x_train, weights, mu, sigma, sinus_type=False)
    
    ## Evaluation 
    y_test = rbf.evaluation_DELTA(x_test, weights, mu, sigma)
    residual_error = np.sum(abs(y_test - square_test))/y_test.shape[0]
    return residual_error, y_test, square_test


def main():
    # GENERATES DATASET (TRAIN & TEST)
    x_train = np.arange(0,2*math.pi,0.1)
    x_test = np.arange(0.05,2*math.pi,0.1)

    # KERNEL PARAMS 
    sigma_vec = [ 0.1, 0.5, 1, 2]
    nr_nodes_vec = [2*math.pi/5]

    # SINUS LEAST SQUARE
    averages = 100
    
    for sigma in sigma_vec:
        test_results = []
        for nr_nodes in nr_nodes_vec:
            residuals_vec = []
            for _ in range(averages):
                mu = np.arange(0,2*math.pi, nr_nodes)
                error, y_test, _ = sinus_LS(x_test, x_train, mu, sigma)
                residuals_vec.append(error)
            error = sum(residuals_vec)/len(residuals_vec)
            std = sum( (residuals_vec - error)**2 )/len(residuals_vec)
            test_results.append([mu.shape[0], error, std])
        print('Results for Least Square (SINUS) sigma: {}'.format(sigma))
        for result in test_results: print('     NODES: {} Residual error: {:0.5f} std. {:.2e}'.format(result[0], result[1], result[2]))

    # SQUARE LEAST SQUARE
    print()
    print('#####################################################')
    averages = 100
    for sigma in sigma_vec:
        test_results = []
        for nr_nodes in nr_nodes_vec:
            residuals_vec = []
            for _ in range(averages):
                mu = np.arange(0,2*math.pi, nr_nodes)
                error, y_test, _ = square_LS(x_test, x_train, mu, sigma)
                residuals_vec.append(error)
            error = sum(residuals_vec)/len(residuals_vec)
            std = sum( (residuals_vec - error)**2 )/len(residuals_vec)
            test_results.append([mu.shape[0], error, std])
        print('Results for Least Square (SQUARE) sigma: {}'.format(sigma))
        for result in test_results: print('     NODES: {} Residual error: {:0.5f} std. {:.2e}'.format(result[0], result[1], result[2]))

    nr_nodes_vec = [0.4]
    sigma_vec = [ 0.1, 0.5, 1, 2]
    # SINUS DELTA-RULE
    print()
    print('#####################################################')
    averages = 100
    for sigma in sigma_vec:
        test_results = []
        for nr_nodes in nr_nodes_vec:
            residuals_vec = []
            for _ in range(averages):
                mu = np.arange(0,2*math.pi, nr_nodes)
                error, y_test, _ = sinus_delta(x_test, x_train, mu, sigma)
                residuals_vec.append(error)
            error = sum(residuals_vec)/len(residuals_vec)
            std = sum( (residuals_vec - error)**2 )/len(residuals_vec)
            test_results.append([mu.shape[0], error, std])
        print('Results for Delta-Rule (SINUS) sigma: {}'.format(sigma))
        for result in test_results: print('     NODES: {} Residual error: {:0.5f} std. {:.2e}'.format(result[0], result[1], result[2]))

    # SQUARE DELTA-RULE
    print()
    print('#####################################################')
    averages = 100
    for sigma in sigma_vec:
        test_results = []
        for nr_nodes in nr_nodes_vec:
            residuals_vec = []
            for _ in range(averages):
                mu = np.arange(0,2*math.pi, nr_nodes)
                error, y_test, _ = square_delta(x_test, x_train, mu, sigma)
                residuals_vec.append(error)
            error = sum(residuals_vec)/len(residuals_vec)
            std = sum( (residuals_vec - error)**2 )/len(residuals_vec)
            test_results.append([mu.shape[0], error, std])
        print('Results for Delta-Rule (SQUARE) sigma: {}'.format(sigma))
        for result in test_results: print('     NODES: {} Residual error: {:0.5f} std. {:.2e}'.format(result[0], result[1], result[2]))



    ######## PLOTING ##########
    VERBOSE=True
    if VERBOSE:
        sigma = 1
        plt.figure('Least Square - Sinus')
        for nr_nodes in nr_nodes_vec:
            mu = np.arange(0,2*math.pi, nr_nodes)
            error, y_test, sinus_test = sinus_LS(x_test, x_train, mu, sigma)
            plt.plot(x_test, y_test, '--', label='Nodes {}'.format(mu.shape[0]))
        plt.plot(x_test, sinus_test, label='True value', c='black')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Least Square - Sinus')
        plt.legend()


        plt.figure('Least Square - Square')
        for nr_nodes in nr_nodes_vec:
            mu = np.arange(0,2*math.pi, nr_nodes)
            error, y_test, square_test = square_LS(x_test, x_train, mu, sigma)
            plt.plot(x_test, y_test, '--', label='Nodes {}'.format(mu.shape[0]))
        plt.plot(x_test, square_test, label='True value', c='black')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Least Square - Square')
        plt.legend()


        sigma = 0.2
        plt.figure('Delta-Rule - Sinus')
        for nr_nodes in nr_nodes_vec:
            mu = np.arange(0,2*math.pi, nr_nodes)
            error, y_test, sinus_test = sinus_delta(x_test, x_train, mu, sigma)
            plt.plot(x_test, y_test, '--', label='Nodes {}'.format(mu.shape[0]))
        plt.plot(x_test, sinus_test, label='True value', c='black')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Delta-Rule - Sinus')
        plt.legend()

        plt.figure('Delta-Rule - Square')
        for nr_nodes in nr_nodes_vec:
            mu = np.arange(0,2*math.pi, nr_nodes)
            error, y_test, square_test = square_delta(x_test, x_train, mu, sigma)
            # y_test = threshold(y_test)
            plt.plot(x_test, y_test, '--', label='Nodes {}'.format(mu.shape[0]))
        plt.plot(x_test, square_test, label='True value', c='black')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Delta-Rule - Square')
        plt.legend()



        plt.show()

if __name__ == '__main__':
    main()

