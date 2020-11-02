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

class RBF():
    def __init__(self,dim,seed=42):
        self.dim=dim
        self.seed=seed
        self.weights=None

    def __str__(self):
        return 'Number of nodes: {} \nseed: {}'.format(self.dim, self.seed)

    def generateData(self,x, noise=False, sigma=0.1):
        if noise:
            sinus = np.sin(2*x) + np.random.randn(x.shape[0])*sigma
            square = signal.square(2*x) + np.random.randn(x.shape[0])*sigma
        else:
            sinus=np.sin(2*x)
            square=signal.square(2*x)
        return sinus,square

    def initWeights(self,sigma=0.1):
        weights = np.random.randn(1, self.dim)*sigma
        self.weights=np.transpose(weights)
        return self.weights
        
    def transferFunction(self,x,mu,sigma):
        PHI = np.zeros((x.shape[0], mu.shape[0]))
        for i in range(x.shape[0]):
            phi = np.exp((-(x[i]-mu)**2)/(2*sigma**2))
            PHI[i,:] = phi
        return PHI

    def activationFunction(self,weights,phi):
        function= phi @ weights
        return function


############# DELTA ###############
    def deltaRule(self, x_train, y_train, weights, phi, eta=0.08):
        # print("Sequantial delta rule")
        
        for i in range(phi.shape[0]):
            phi_tmp = phi[i,:].reshape((phi[i,:].shape[0],1))
            weights += eta*(y_train[i] - (np.transpose(phi_tmp)@weights)[0][0])*phi_tmp
   
        return weights

    def train_DELTA(self, x_train, weights, mu, sigma, eta, sinus_type=True, noise=True):
        phi = self.transferFunction(x_train,mu,sigma)

        # Generate the first sinus with error.
        if sinus_type:
                y_train, _ = self.generateData(x_train, noise=noise)
        else:
            _, y_train = self.generateData(x_train, noise=noise)
        y_train = y_train.reshape((y_train.shape[0],1))

        # Get the first error with randomnized weights
        output = self.evaluation_DELTA(x_train, weights, mu, sigma)
        error_vec = [np.sum(abs(output - y_train))/y_train.shape[0]]

        # Setting up the while-loop
        delta_error = 1
        epoch_vec = [1]
        epoch = 1
        while abs(delta_error) > 0.001:
            epoch += 1
            epoch_vec.append(epoch)
            # Update weights with Delta-Rule
            weights = self.deltaRule(x_train, y_train, weights, phi, eta=eta)
            output = self.evaluation_DELTA(x_train, weights, mu, sigma)

            # Stores the Residual-error and takes delta error for convergens
            error_vec.append(np.sum(abs(output - y_train))/y_train.shape[0])
            delta_error = abs(error_vec[-2] - error_vec[-1])

            # Generate the next two periods of data. 
            if sinus_type:
                y_train, _ = self.generateData(x_train, noise=noise)
            else:
                _, y_train = self.generateData(x_train, noise=noise)
            y_train = y_train.reshape((y_train.shape[0],1))


        return weights, error_vec, epoch_vec, epoch

    def evaluation_DELTA(self, xtest, weights, mu, sigma):
        phi=self.transferFunction(xtest,mu,sigma)
        return self.activationFunction(weights, phi)

def sinus_delta(x_test, x_train, mu, sigma, eta):
    dim=mu.shape[0]
    rbf = RBF(dim)
    
    ## Generate data
    sinus, _   = rbf.generateData(x_train, noise=True)
    sinus_test, _ = rbf.generateData(x_test, noise=True)
    sinus_test = sinus_test.reshape((sinus_test.shape[0],1))
    ## Init and train.
    weights         = rbf.initWeights()
    weights, _, _,epoch  = rbf.train_DELTA(x_train, weights, mu, sigma, eta=eta)

    ## Evaluation 
    y_test = rbf.evaluation_DELTA(x_test, weights, mu, sigma)
    tmp = abs(y_test - sinus_test)
    residual_error = np.sum(abs(y_test - sinus_test))/y_test.shape[0]
    return residual_error, y_test, sinus_test, epoch

def main():
    # GENERATES DATASET (TRAIN & TEST)
    x_train = np.arange(0,2*math.pi,0.1)
    x_test = np.arange(0.05,2*math.pi,0.1)

    # KERNEL PARAMS 
    sigma = 0.5
    nr_nodes = 0.2


    averages = 200
    test_results = []
  
    eta_vec = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    for eta in eta_vec:
        residuals_vec = []
        epoch_vec = []
        for _ in range(averages):
            mu = np.arange(0,2*math.pi, nr_nodes)
            error, y_test, _, epoch = sinus_delta(x_test, x_train, mu, sigma, eta)
            residuals_vec.append(error)
            epoch_vec.append(epoch)
        error = sum(residuals_vec)/len(residuals_vec)
        std = sum( (residuals_vec - error)**2 )/len(residuals_vec)
        test_results.append([eta, error, std, sum(epoch_vec)/averages])
    print('Results for Delta-Rule (SINUS) sigma: {}'.format(sigma))
    for result in test_results: print('     ETA: {} Residual error: {:0.5f} std. {:.2e} avr. epochs {}'.format(result[0], result[1], result[2], result[3]))


if __name__ == "__main__":
    main()