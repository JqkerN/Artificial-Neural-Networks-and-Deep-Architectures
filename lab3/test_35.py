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

def randomPatterns(nPatterns, size):
    rndPatterns=np.random.choice([-1,1],[nPatterns,size])
    # rndPatterns=np.sign(0.5+np.random.randn(nPatterns,size))
    return rndPatterns




def main():

    # patterns = loadData('pict.dat')     # Pattern 1-11
    p=30
    patterns=randomPatterns(nPatterns=p, size=100)
    # print(patterns.shape)
    patterns_1_3 = [patterns[index,:].reshape(1,100) for index in range(p)]
    # patterns_4_11 = [patterns[3+index,:].reshape(1,1024) for index in range(8) ]
    # print(len(patterns_1_3))
    diag=[True, False]
    for h in range(2):
        totvec=[]
        for k in range(len(patterns_1_3)):
            totCorrect=0
            patterns_t=patterns_1_3[0:k+1]
            network = RNN(size=100, sequential=False, random=False, diagonal=diag[h])
            network.init_weights(patterns_t)
            averages=5
            for i, pattern in enumerate(patterns_t):
                nCorrect=0
                print(h," ",k," ",i)
                print()
                OGpattern=pattern.copy()
                for j in range(averages):
                    patternD=distort(OGpattern,15)
                    x_output = network.train(patternD)
                    nCorrect += ((np.count_nonzero(x_output==OGpattern))/patternD.shape[1])*100
            
                nCorrect=nCorrect/averages
            
                if int(nCorrect==100):
                    totCorrect += 1
            totCorrect=totCorrect/len(patterns_t)*100
            totvec.append(totCorrect)
        


        plt.plot(np.arange(0,p,1),totvec, label=("Diagonal " + str(diag[h])))
        # plt.imshow(x_output.reshape(32,32), cmap='gray')

    plt.legend()
    plt.show()

    # plt.plot(noises,nCorrect, label=("Pattern " + str(i+1)))

    # plt.axis(xmin=0,xmax=100, ymin=0, ymax=100)
    # plt.xlabel("Noise [%]")
    # plt.ylabel("Accuracy [%]")
    # plt.title("Capacity")
    # plt.show()


if __name__ == "__main__":
    main()

