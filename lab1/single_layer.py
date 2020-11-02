

import numpy as np
import matplotlib.pyplot as plt


class perceptron():

    def __init__(self, layers, bias=True, batch=True, learningRate=0.001, seed=42):
        self.layers = layers
        self.bias = bias
        self.batch = batch
        self.classData = None
        self.T = None
        self.classA = None
        self.classB = None
        self.W = None
        self.learningRate = learningRate
        self.seed = seed
        self.outputs = None
        return

    def generateClassData(self, nA, nB, mA, mB, sigmaA, sigmaB):

        classA_size = mA.size
        self.classA = np.random.RandomState(seed=self.seed).randn(
            classA_size, nA)*sigmaA + np.transpose(np.array([mA]*nA))

        classB_size = mB.size
        self.classB = np.random.RandomState(seed=self.seed).randn(
            classB_size, nB)*sigmaB + np.transpose(np.array([mB]*nB))

        classAB = np.concatenate((self.classA, self.classB), axis=1)
        # X in math
        if self.bias:
            classData = np.concatenate((classAB, np.ones((1, nA+nB))), axis=0)
        else:
            classData = classAB

        T = np.concatenate((np.ones((1, nA)), np.ones((1, nB))*(-1)), axis=1)

        shuffler = np.random.RandomState(seed=self.seed).permutation(nA+nB)
        self.classData = classData[:, shuffler]
        self.T = T[:, shuffler]

        return

    def plotData(self, name="Decision boundary", epoch=None):
        plt.figure(name)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.scatter(self.classA[0, :], self.classA[1, :], c="red")
        plt.scatter(self.classB[0, :], self.classB[1, :], c="blue")
        y1 = -self.W[0, 2]/self.W[0, 1] - self.W[0, 0]/self.W[0, 1]*(-2)
        y2 = -self.W[0, 2]/self.W[0, 1] - self.W[0, 0]/self.W[0, 1]*(2)
        plt.plot([-2, 2], [y1, y2])
        plt.axis([-2, 2, -2, 2])
        plt.title("Epoch: {}, Batch: {}".format(epoch+1, self.batch))
        plt.draw()
        plt.pause(0.001)
        return

    def initWeights(self, dim=2, sigma=1):
        if self.bias:
            dim += 1
        self.W = np.random.RandomState(seed=self.seed).randn(1, dim)*sigma

        return

    def deltaRule(self):
        if self.batch:
            deltaW = -self.learningRate * \
                np.dot((np.dot(self.W, self.classData) - self.T),
                       np.transpose(self.classData))
            self.W += deltaW
        else:
            deltaW = np.zeros(np.shape(self.W))
            for i in range(np.shape(self.classData)[1]):
                deltaW = -self.learningRate * \
                    (np.dot(
                        self.W, self.classData[:, i]) - self.T[:, i])*np.transpose(self.classData[:, i])
                self.W += deltaW

    def activationFunction(self, prediction):
        if self.batch:
            for index, p in enumerate(prediction[0]):
                if p > 0:
                    prediction[0, index] = 1
                else:
                    prediction[0, index] = -1
        else:
            if prediction > 0:
                prediction = 1
            else:
                prediction = -1
        return prediction

    def perceptronLearning(self):
        if self.batch:
            deltaW = -self.learningRate * \
                np.dot((self.activationFunction(np.dot(self.W, self.classData)) - self.T),
                       np.transpose(self.classData))
            self.W += deltaW
        else:
            deltaW = np.zeros(np.shape(self.W))
            for i in range(np.shape(self.classData)[1]):
                deltaW = -self.learningRate * \
                    (self.activationFunction(np.dot(
                        self.W, self.classData[:, i])) - self.T[:, i]) * np.transpose(self.classData[:, i])
                self.W += deltaW

    def train(self, epochs=50, verbose=True, method=1):
        # Method 1 = Perceptron Learning
        # Method 2 = Delta Rule
        epoch_vec = []
        loss_vals = []
        for i in range(epochs):
            if method == 1:
                self.perceptronLearning()
            elif method == 2:
                self.deltaRule()
            if verbose:
                self.plotData(epoch=i)
            epoch_vec.append(i)
            loss_vals.append(self.evaluation())
        if verbose:
            plt.savefig("images/Decision boundary")
            self.plotData(
                name="Decision boundary - epochs ({})".format(i+1), epoch=i)
            plt.savefig("images/Decision boundary - epochs ({})".format(i+1))
        return epoch_vec, loss_vals

    def classify(self, dataset=None):
        try:
            if dataset == None:
                dataset = self.classData
        except:
            outputs = np.dot(self.W, dataset)
            for index, output in enumerate(outputs[0]):
                if output >= 0:
                    outputs[0, index] = 1
                else:
                    outputs[0, index] = -1
            self.outputs = outputs

    def evaluation(self, dataset=None):
        if dataset == None:
            dataset = self.classData
        self.classify(dataset)
        loss_vec = self.outputs == self.T
        nr_trues = np.count_nonzero(loss_vec)
        loss_val = 1 - nr_trues/np.shape(self.T)[1]
        return loss_val


def main():
    # Parameters for generate_data
    mA = np.array([1, 0.5])
    mB = np.array([-1, 0])
    nA = 100
    nB = 100
    sigmaA = 0.4
    sigmaB = 0.4

    single_layer = perceptron(1, batch=False, learningRate=0.001, seed=42)
    single_layer.generateClassData(nA, nB, mA, mB, sigmaA, sigmaB)
    single_layer.initWeights()
    epoch_vec, loss_vals = single_layer.train(
        method=2, verbose=True, epochs=10)

    # SEQ
    ETA = [0.001, 0.002, 0.004]
    legends = []
    for eta in ETA:
        plt.figure("Learning_Curve_seq")
        single_layer = perceptron(1, batch=False, learningRate=eta, seed=42)
        single_layer.generateClassData(nA, nB, mA, mB, sigmaA, sigmaB)
        single_layer.initWeights()
        epoch_vec, loss_vals = single_layer.train(
            method=2, verbose=False, epochs=10)
        plt.plot(epoch_vec, [100*i for i in loss_vals], "-.")
        legends.append("Learning rate = " + str(eta))
    plt.title("Learning Curve, Sequential")
    plt.xlabel("Epoch")
    plt.ylabel("Missclassified [%]")
    plt.legend(legends)
    plt.savefig("images/Learning_Curve_seq")

    # BATCH
    ETA = [0.001, 0.002, 0.004]
    legends = []
    for eta in ETA:
        plt.figure("Learning_Curve_batch")
        single_layer = perceptron(1, batch=True, learningRate=eta, seed=42)
        single_layer.generateClassData(nA, nB, mA, mB, sigmaA, sigmaB)
        single_layer.initWeights()
        epoch_vec, loss_vals = single_layer.train(
            method=2, verbose=False, epochs=10)
        plt.plot(epoch_vec, [100*i for i in loss_vals], "-.")
        legends.append("Learning rate = " + str(eta))
    plt.title("Learning Curve, Batch")
    plt.xlabel("Epoch")
    plt.ylabel("Missclassified [%]")
    plt.legend(legends)
    plt.savefig("images/Learning_Curve_batch")


if __name__ == "__main__":
    main()
