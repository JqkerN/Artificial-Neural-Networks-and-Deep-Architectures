import numpy as np
import random
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)


class Node():
    def __init__(self):
        self.weights = np.random.uniform(0, 1, 31)


class SOM():

    def __init__(self, eta=0.2):
        self.nodes = np.array([[None]*10]*10)
        self.index = None
        self.eta = eta

    def __str__(self):
        return str(self.nodes.shape)

    def initWeights(self):
        for i in range(10):
            for j in range(10):
                self.nodes[i][j] = Node()

    def euclidianDist(self, pattern):
        dBest = 100000000
        for i in range(10):
            for j in range(10):
                d = np.linalg.norm(pattern-self.nodes[i][j].weights)
                if d < dBest:
                    dBest = d
                    iBest = i
                    jBest = j
        return iBest, jBest

    def neighbourhood(self, iBest, jBest, epoch, epochs):
        if epoch/epochs <= 0.2:
            dist = 2
        elif epoch/epochs <= 0.5:
            dist = 1
        else:
            dist = 0

        hood = []
        iHood = np.linspace(iBest-dist, iBest+dist, 2*dist+1)
        jHood = np.linspace(jBest-dist, jBest+dist, 2*dist+1)

        for i in iHood:
            for j in jHood:
                if abs(i-iBest) + abs(j-jBest) > dist or 0 > i or i > 9 or 0 > j or j > 9:
                    continue
                else:
                    hood.append([i, j])
        hood = np.array(hood)
        return hood

    def weightsUpdate(self, pattern, hood):
        for indexes in hood:
            i = int(indexes[0])
            j = int(indexes[1])
            self.nodes[i][j].weights = self.nodes[i][j].weights + self.eta * \
                np.subtract(pattern, self.nodes[i][j].weights)


def main():

    ####### Import vote data ##############
    data = []
    with open('/home/andrej/school/ann-course/lab2/votes.dat', 'r') as f:
        d = f.readlines()
        for i in d:
            k = i.rstrip().split(",")
            data.append([float(i) for i in k])
    data = np.array(data, dtype='O')
    voteData = np.reshape(data, (349, 31))
    ######################################

    #  Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
    # Use some color scheme for these different groups

    data = []
    with open('/home/andrej/school/ann-course/lab2/mpparty.dat', 'r') as f:
        d = f.readlines()
        for i in d:
            k = i.rstrip().split(",")
            data.append([int(i) for i in k])
    partyData = np.array(data, dtype='O')

    ####### init som and weights #########
    som = SOM()
    som.initWeights()
    epochs = 20
    ######################################

    ######## Training ###################
    for epoch in range(epochs):
        shuffler = np.random.permutation(349)
        for i in shuffler:
            iBest, jBest = som.euclidianDist(voteData[i][:])
            hood = som.neighbourhood(iBest, jBest, epoch, epochs)
            weights = som.weightsUpdate(voteData[i][:], hood)
        som.eta *= 0.95
    # ######################################

    # ########## Testing ##################
    winnerIndexes = []
    for i in range(349):
        iBest, jBest = som.euclidianDist(voteData[i][:])
        winnerIndexes.append((iBest, jBest))
    # ######################################

    plt.figure('Party member comparison')
    plt.title('Party Member comparison')
    # plt.grid()
    nr_attributes = 8
    grid = np.array([[[0]*10]*10]*nr_attributes)

    index = 0
    for winner in winnerIndexes:
        partyBelonging = partyData[index]
        grid[partyBelonging[0]][winner[0]][winner[1]] += 1
        index += 1

    color = ['k', '#0000ff', '#3399ff', '#ff0000',
             '#990000', '#339933', '#6600cc', '#49ff33']
    party = ['-', 'M', 'FP', 'S', 'V', 'MP', 'KD', 'C']
    for i in range(10):
        for j in range(10):
            bestNr = 0
            bestAtt = None
            for attribute in range(nr_attributes):
                nr = grid[attribute][i][j]
                if nr > bestNr:
                    bestNr = nr
                    bestAtt = attribute
            if bestAtt == None:
                continue
            else:
                plt.scatter(i, j, color=color[bestAtt], s=2500)
                plt.text(
                    i, j, party[bestAtt], horizontalalignment='center', verticalalignment='center', fontsize=14)
    plt.axes([0, 10, 0, 10])

    # GENDER
    data = []
    with open('/home/andrej/school/ann-course/lab2/mpsex.dat', 'r') as f:
        d = f.readlines()
        for i in d:
            k = i.rstrip().split(",")
            data.append([int(i) for i in k])
    partyData = np.array(data, dtype='O')

    plt.figure('Gender Comparision')
    plt.title('Gender Comparision')
    nr_attributes = 2
    grid = np.array([[[0]*10]*10]*nr_attributes)

    index = 0
    for winner in winnerIndexes:
        partyBelonging = partyData[index]
        grid[partyBelonging[0]][winner[0]][winner[1]] += 1
        index += 1

    color = ['blue', 'pink']
    # % Coding: Male 0, Female 1

    party = ['M', 'F']
    for i in range(10):
        for j in range(10):
            bestNr = 0
            bestAtt = None
            for attribute in range(nr_attributes):
                nr = grid[attribute][i][j]
                if nr > bestNr:
                    bestNr = nr
                    bestAtt = attribute
            if bestAtt == None:
                continue
            else:
                plt.scatter(i, j, color=color[bestAtt], s=2500)
                plt.text(
                    i, j, party[bestAtt], horizontalalignment='center', verticalalignment='center', fontsize=14)
    plt.axes([0, 10, 0, 10])
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
