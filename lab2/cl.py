from rbf import RBF
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math


#! Cl Class ONLY FOR 1-D DATA
class CL():
    def __init__(self,nUnits,data,width,steps,learningRate,show=False, info=True, winners=1):
        self.nUnits=nUnits
        self.weights=None
        self.data=data
        self.nDataPoints=data.shape[0]
        self.trainingData=None
        self.width=width
        self.steps=steps
        self.learningRate=learningRate
        self.show=show
        self.step=None
        self.winners=np.zeros(winners,dtype=int)
        if self.show:
            self.y=np.zeros(nUnits)
            plt.show()
        if info:
            print(self)
        

    def __str__(self):
        return 'CL class: \n Units: {} \n Datapoints: {} \n Dimensions: {} \n Winners: {}'.format(self.nUnits, self.nDataPoints, self.data.shape, self.winners.shape[0])
    
    def initWeights(self):
        weights = np.random.uniform(0,1, [self.data.shape[1],self.nUnits])* np.amax(self.data)
        self.weights=np.transpose(weights)

    def trainingVector(self):
        index=np.random.randint(0,self.nDataPoints)
        self.trainingData=self.data[index]

    def selection(self):

        distance=np.zeros(self.weights.shape[0])
        for index, weight in enumerate(self.weights):
            distance[index] = np.linalg.norm(self.trainingData-weight)
        
        distance=np.ndarray.tolist(distance)

        for i in range(self.winners.shape[0]):
            winnerValue =np.amin(distance)    
            self.winnerIndex = np.where(distance == winnerValue)[0][0]
            self.winners[i]=int(self.winnerIndex)
            distance.pop(self.winnerIndex)

    def update(self):

        for winner in self.winners:
            self.weights[winner]=self.weights[winner]+self.learningRate*(self.trainingData-self.weights[winner])


    def train(self):
        self.initWeights()
        for self.step in range(self.steps):
            self.trainingVector()
            self.selection()
            self.update()
        if self.show:
            self.plot()

    
    def plot(self):
        plt.title("RBF centers")
        weights = self.weights
        if self.data.shape[1]==1:
            y=np.zeros(self.nUnits)
        else:
            y=weights[:,1]
            plt.scatter(self.data[:,0],self.data[:,1],c="r")
        plt.scatter(weights[:,0],y,c="b")
        plt.legend(["Data","Weights"])
        # plt.axis(xmin=0,xmax=6.5)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()







