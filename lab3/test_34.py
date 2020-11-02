from RNN import RNN
import numpy as np
import math
from mpl_toolkits import mplot3d
import itertools
import matplotlib.pyplot as plt
from test_33 import loadData

def distort(pattern,noise):
    patternD=pattern.copy()
    patternD=np.squeeze(patternD)
    size=patternD.shape[0]
    noise=int(np.floor(noise/100*size))
    index=np.random.randint(0,size,noise)
    distorted=np.where(patternD[index]==1, -1, 1)
    patternD[index]=distorted
    patternD=np.reshape(patternD,[1,patternD.shape[0]])
    return patternD

    



def main():

    patterns = loadData('pict.dat')     # Pattern 1-11
    patterns_1_3 = [patterns[index,:].reshape(1,1024) for index in range(3) ]
    patterns_4_11 = [patterns[3+index,:].reshape(1,1024) for index in range(8) ]
    

    network = RNN(size=1024, sequential=False, random=False)
    network.init_weights(patterns_1_3)
    noises=np.arange(0,100,5)
    averages=1000
    for i, pattern in enumerate(patterns_1_3):
        OGpattern=pattern.copy()
        nCorrect=np.zeros((noises.shape[0],1))
        for k, noise in enumerate(noises):
            for j in range(averages):
                patternD=distort(OGpattern,noise)
                x_output = network.train(patternD)
                nCorrect[k][0] += ((np.count_nonzero(x_output==OGpattern))/patternD.shape[1])*100

        nCorrect=nCorrect/averages
        plt.plot(noises,nCorrect, label=("Pattern " + str(i+1)))
        
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

