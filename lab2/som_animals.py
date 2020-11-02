import numpy as np
import random


class SOM():

    def __init__(self, nNodes, inputDim, nClass, eta=0.2):
        self.nNodes = nNodes
        self.weights = None
        self.eta = eta
        self.inputDim = inputDim
        self.nClass = nClass

    def initWeights(self):
        self.weights = np.zeros((self.nNodes, self.inputDim))
        for i in range(self.nNodes):
            self.weights[i][:] = np.random.random(self.inputDim)
        return self.weights

    def euclidianDist(self, pattern):
        dBest = 10000
        for i in range(self.weights.shape[0]):
            d = np.transpose(
                pattern-self.weights[i][:])@(pattern-self.weights[i][:])
            if d < dBest:
                dBest = d
                iBest = i
        return iBest

    def neighbourhood(self, index, epoch, epochs):
        if epoch/epochs <= 0.1:
            dist = 25
        elif epoch/epochs <= 0.25:
            dist = 10
        elif epoch/epochs <= 0.75:
            dist = 5
        else:
            dist = 1

        neighbours = np.linspace(
            index-dist, index+dist, 2*dist+1)
        neighbours = np.where(neighbours < 0, neighbours + 100, neighbours)
        neighbours = np.where(neighbours > 99, neighbours - 100, neighbours)
        return neighbours

    def weightsUpdate(self, pattern, neighbours):
        for i in neighbours:
            self.weights[int(i)][:] = self.weights[int(i)][:] + \
                self.eta*(pattern-self.weights[int(i)][:])
        return self.weights


def main():

    ######## Import animal data ############
    data = []
    with open('/home/andrej/school/ann-course/lab2/animals.dat', 'r') as f:
        d = f.readlines()
        for i in d:
            k = i.rstrip().split(",")
            data.append([int(i) for i in k])
    data = np.array(data, dtype='O')
    animalData = np.reshape(data, (32, 84))

    ######### Import animal names ############
    data = []
    with open('/home/andrej/school/ann-course/lab2/animalnames.txt', 'r') as f:
        d = f.readlines()
        for i in d:
            k = i.rstrip("'").split()
            data.append([i for i in k])
    data = np.array(data, dtype='O')
    animalNames = data
    animalNames = np.squeeze(animalNames)
    ########################################

    ####### init som and weights ###########
    som = SOM(nNodes=100, inputDim=84, nClass=32)
    weights = som.initWeights()
    epochs = 20
    ######################################

    ######## Training ###################
    for epoch in range(epochs):
        for i in range(32):
            iBest = som.euclidianDist(animalData[i][:])
            # print(iBest)
            neighbours = som.neighbourhood(iBest, epoch, epochs)
            # print(neighbours)
            som.weightsUpdate(animalData[i][:], neighbours)
    ######################################

    ########## Testing ##################
    winnerIndexes = []
    for i in range(32):
        iBest = som.euclidianDist(animalData[i][:])
        winnerIndexes.append(iBest)
    ######################################

    animalNames = np.ndarray.tolist(animalNames)
    animalNames = [x for _, x in sorted(zip(winnerIndexes, animalNames))]
    print(animalNames)
    winnerIndexes = sorted(winnerIndexes)
    # print(winnerIndexes)


if __name__ == "__main__":
    main()
