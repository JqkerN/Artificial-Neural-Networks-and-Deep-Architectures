from rbf import RBF
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math



def threshold(dataset,threshold=0):
    thresholded = [1 if datapoint > threshold else -1 for datapoint in dataset]
    thresholded=np.array(thresholded)
    print("Output data thresholded")
    return thresholded


def residualError(dataset,trueData):
    residual_error = np.sum(abs(dataset - trueData))/dataset.shape[0]
    print("Residual Error: ", residual_error)
    return residual_error

def main():
    ## generate data and define inputs
    mu = np.arange(0,2*math.pi,0.1)
    sigma = 0.1
    x_train = np.arange(0,2*math.pi,0.1)
    x_test = np.arange(0.05,2*math.pi,0.1)

    # #! DELTA RULE

    #! LEAST SQUARE
    ## init rbf class
    dim=mu.shape[0]
    rbf_LS=RBF(dim)

    ## Generate data
    sinus, square   = rbf_LS.generateData(x_train)
    sinus_test, square_test = rbf_LS.generateData(x_test)

    ## Init and train.
    weights         = rbf_LS.initWeights()
    weights, error  = rbf_LS.train_DELTA(x_train, square, weights, mu, sigma)
    
    ## Evaluation 
    print(rbf_LS)
    y_test = rbf_LS.evaluation_DELTA(x_test, weights, mu, sigma)
    # y_test=threshold(y_test)
    re=residualError(y_test,square_test)
    plt.figure('Least Square Error')
    plt.plot(x_test, y_test, label='Approximation')
    plt.plot(x_test, square_test, label='True value')
    plt.title('Least Square Error')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()