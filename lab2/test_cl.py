from rbf import RBF
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
from cl import CL


def generateData(noisy,sigma):
    data = np.arange(0,2*math.pi,0.2)
    if noisy:
        data=data+np.random.randn(data.shape[0])*sigma
    data=np.reshape(data,(data.shape[0],1))
    return data


def readData(file):
    inputData = np.array([])
    with open(file, 'r') as f:
        d = f.readlines()
        index=0
        for i in d:
            k = i.rstrip().split("\t")
            k=([i.split(" ") for i in k])
            if index==0:
                inputData = np.array([float(i) for i in k[0]])
                outputData = np.array([float(i) for i in k[1]])
            else:
                inputData=np.vstack((inputData,([float(i) for i in k[0]])))
                outputData=np.vstack((outputData,([float(i) for i in k[1]])))
            index=1
    return inputData, outputData




def main():
    # inputData=generateData(noisy=False,sigma=0.1)
    inputData,outputData=readData('ballist.dat')
    cl=CL(50,inputData,1,2000,0.2,show=True,winners=2)
    cl.train()


    # plt.scatter(cl.weights[:,0],cl.weights[:,1],c="b")
    # plt.scatter(inputData[:,0],inputData[:,1],c="r")
    # plt.show()

    

if __name__ == "__main__":
    main()

    