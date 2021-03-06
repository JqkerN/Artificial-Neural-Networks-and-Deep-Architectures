
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

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

    def train_DELTA(self, x_train, weights, mu, sigma, sinus_type=True, noise=True):
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
        while abs(delta_error) > 0.01:
            epoch += 1
            epoch_vec.append(epoch)
            # Update weights with Delta-Rule
            weights = self.deltaRule(x_train, y_train, weights, phi)
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


        return weights, error_vec, epoch_vec

    def evaluation_DELTA(self, xtest, weights, mu, sigma):
        phi=self.transferFunction(xtest,mu,sigma)
        return self.activationFunction(weights, phi)



############### LS ################
    def leastSquares(self, PHI, function):
        # rest=trainingData%batchSize
        # batches = [trainingData[i*batchSize:(i+1)*batchSize] for i in range(int(trainingData.shape[0]/batchSize))]
        # batches.append(trainingData[-rest,-1])
        self.weights = np.linalg.lstsq(PHI, function)
        return self.weights[0],self.weights[1]
        
    def train_LS(self, x_train, y_train, weights, mu, sigma):
        phi = self.transferFunction(x_train,mu,sigma)
        weights, error = self.leastSquares(phi, y_train)
        return weights, error

    def evaluation_LS(self, xtest, weights, mu, sigma):
        phi=self.transferFunction(xtest,mu,sigma)
        return self.activationFunction(weights,phi)

    def classify(self):

        pass

def main():
    ## generate data and define inputs
    mu = np.arange(0,2*math.pi,0.1)
    sigma = 0.1
    x_train = np.arange(0,2*math.pi,0.1)
    x_test = np.arange(0.05,2*math.pi,0.1)

    # #! DELTA RULE
    ## init rbf class
    dim=mu.shape[0]
    rbf_delta=RBF(dim)
    print(rbf_delta)

    ## Generate data
    sinus, square   = rbf_delta.generateData(x_train)
    sinus_test, square_test = rbf_delta.generateData(x_test)

    ## Init and train.
    weights         = rbf_delta.initWeights()
    weights, error_vec, epoch_vec  = rbf_delta.train_DELTA(x_train, weights, mu, sigma, sinus_type=True)
    
    plt.figure('Training Curve')
    plt.plot(epoch_vec, error_vec)
    plt.title('Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Residual Error')

    ## Evaluation 
    y_test = rbf_delta.evaluation_DELTA(x_test, weights, mu, sigma)
    plt.figure('Delta Rule')
    plt.plot(x_test, y_test, label='Approximation')
    plt.plot(x_test, sinus_test, label='True value')
    plt.title('Delta Rule')
    plt.legend()

    #! LEAST SQUARE
    ## init rbf class
    dim=mu.shape[0]
    rbf_LS=RBF(dim)
    print(rbf_LS)

    ## Generate data
    sinus, square   = rbf_LS.generateData(x_train)
    sinus_test, square_test = rbf_LS.generateData(x_test)

    ## Init and train.
    weights         = rbf_LS.initWeights()
    weights, error  = rbf_LS.train_LS(x_train, square, weights, mu, sigma)
    # print('Error (lstsq): ', error[0])
    
    ## Evaluation 
    y_test = rbf_LS.evaluation_LS(x_test, weights, mu, sigma)
    plt.figure('Least Square Error')
    plt.plot(x_test, y_test, label='Approximation')
    plt.plot(x_test, square_test, label='True value')
    plt.title('Least Square Error')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
