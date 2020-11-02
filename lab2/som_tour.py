import numpy as np
import random
import matplotlib.pyplot as plt


class SOM():

    def __init__(self, nNodes, inputDim, nClass, eta=0.85):  # eta scrambled input = 1.1
        self.nNodes = nNodes
        self.weights = None
        self.eta = eta
        self.inputDim = inputDim
        self.nClass = nClass

    def initWeights(self):
        self.weights = np.zeros((self.nNodes, self.inputDim))
        for i in range(self.nNodes):
            self.weights[i][:] = np.random.uniform(0, 1, self.inputDim)
        return self.weights

    def euclidianDist(self, pattern):
        dBest = 100000000
        iBest = 0
        for i in range(self.weights.shape[0]):
            d = np.linalg.norm(pattern-self.weights[i, :])
            if d < dBest:
                dBest = d
                iBest = i
        return iBest

    def neighbourhood(self, index, epoch, epochs):
        if epoch/epochs <= 0.2:
            dist = 2
        elif epoch/epochs <= 0.5:
            dist = 1
        else:
            dist = 0

        neighbours = np.linspace(
            index-dist, index+dist, 2*dist+1)
        neighbours = np.where(neighbours < 0, neighbours + 10, neighbours)
        neighbours = np.where(neighbours > 9, neighbours - 10, neighbours)
        return neighbours

    def weightsUpdate(self, pattern, neighbours):
        for i in neighbours:
            self.weights[int(i)][:] = self.weights[int(i)][:] + \
                self.eta*np.subtract(pattern, self.weights[int(i)][:])
        return self.weights


def main():

    cityData = np.array([[0.4000, 0.4439], [0.2439, 0.1463], [0.1707, 0.2293],  [0.2293, 0.7610], [0.5171, 0.9414], [
                        0.8732, 0.6536], [0.6878, 0.5219], [0.8488, 0.3609], [0.6683, 0.2536], [0.6195, 0.2634]])

    plt.figure('map')
    xData = []
    yData = []
    for data in cityData:
        xData.append(data[0])
        yData.append(data[1])
    plt.scatter(xData, yData, c='red', s=50)

    ####### init som and weights ###########
    som = SOM(nNodes=10, inputDim=2, nClass=10)
    weights = som.initWeights()
    epochs = 80
    ######################################

    ######## Training ###################
    for epoch in range(epochs):
        shuffler = np.random.permutation(10)
        datapoint = np.arange(0, 10, 1)
        datapoint = datapoint[shuffler]
        for i in datapoint:
            iBest = som.euclidianDist(cityData[i][:])
            neighbours = som.neighbourhood(iBest, epoch, epochs)
            weights = som.weightsUpdate(cityData[i][:], neighbours)
        som.eta *= 0.95
    # ######################################

    # ########## Testing ##################
    winnerIndexes = []
    for i in range(10):
        iBest = som.euclidianDist(cityData[i][:])
        # print(iBest)
        winnerIndexes.append(iBest)
    # ######################################

    # Make tour circluar
    # winnerIndexes.append(winnerIndexes[0])
    winnerIndexes = sorted(winnerIndexes)
    winnerIndexes = np.array(winnerIndexes)
    print(winnerIndexes)

    winnerIndexes = np.argsort(winnerIndexes)
    winnerIndexes = np.hstack((winnerIndexes, winnerIndexes[0]))

    print(winnerIndexes)

    X = [cityData[i, 0] for i in winnerIndexes]
    Y = [cityData[i, 1] for i in winnerIndexes]
    plt.plot(X, Y, linewidth=2, color='black')

    for data in weights:
        plt.scatter(data[0], data[1], c='blue', s=50)
    plt.title('Cyclic Tour')
    plt.legend(['Suggested tour', 'Actual cities', 'Weights'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([0, 1.1, 0, 1.1])
    plt.show()


if __name__ == "__main__":
    main()
