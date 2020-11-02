import matplotlib.pyplot as plt
import numpy as np
import math
from generateData import generateClassData
from progressBar import progressBar

class neuralNetwork():
    def __init__(self, layers, bias=True):
        """
        hiddenLayers: Number of hiddenLayers [neurons for lvl1, ... etc]\n
        bias: True/False\n
        seed: seed number\n
        """
        self.layers = layers
        self.numberOfLayers = len(layers)
        self.weights = []
        self.bias = bias


    def __str__(self):
        """
        What information that should be shown about the NN is stated here.
        Structure: the structure of the NN.
        Bias: True/False
        """
        structure = "structure : {} \n".format(
            [np.shape(np.transpose(w)) for w in self.weights])
        layers = "Layers (neurons): {} \n".format(
            self.layers)

        bias = "Bias: {} \n".format(self.bias)
        return structure + layers + bias

    def initWeights(self, dim=2, sigma=1):
        """
        dim: The dimension of the input layer\n
        sigma: Default value 0.1.
        """
        # Init weights for hidden layers
        for layer in self.layers:
            if self.bias:
                dim += 1
            self.weights.append(np.random.randn(layer, dim)*sigma)
            dim = layer


    def transferFunction(self, x):
        return 2 / (1 + np.exp(-x)) - 1

    def activationFunction(self, data):
        data[data > 0] = 1
        data[data <= 0] = -1
        return data

    def forwardpass(self, data):
        if self.bias:
            patterns = np.concatenate((data, np.ones((1, np.shape(data)[1]))), axis=0)
        else:
            patterns = data
        hin = self.weights[0] @ patterns

        hout = self.transferFunction(hin)
        if self.bias:
            hout =  np.concatenate((hout, np.ones((1, np.shape(data)[1]))), axis=0)

        oin = self.weights[1] @ hout
        out = self.transferFunction(oin)
        return out

    def classify(self, data):
        data = self.forwardpass(data=data)
        return self.activationFunction(data)

    def eval(self, data, targets, verbose=False):
        classified_data = self.classify(data)
        accuracy = np.count_nonzero(classified_data == targets)/np.shape(targets)[1]*100
        print("Accuracy: ", accuracy)
        if verbose:
            plt.figure("Decision Boundary")
            plt.scatter(data[0, np.where(classified_data==targets)], data[1, np.where(classified_data==targets)], c="green")
            plt.scatter(data[0, np.where(classified_data!=targets)], data[1, np.where(classified_data!=targets)], c="red")
            

    def loss_val(self, data, target):
        loss = 1 / (2*np.shape(target)[1]) * np.sum( np.power(self.forwardpass(data=data) - target, 2))
        return loss

    def train(self, x_train, y_train, x_valid, y_valid, epochs, eta=0.001, alpha=0.9):
        def forwardpass(x_train):
            """
            Description:\n
                Forwardpass function (recursive function)\n
            Input:\n
                x_train: the intput x_train for current layer
                layer: current layer (number)
                out_vec: the output vector with corresponding output
            """
            if self.bias:
                patterns = np.concatenate((x_train, np.ones((1, np.shape(x_train)[1]))), axis=0)
            else:
                patterns = x_train
            hin = self.weights[0] @ patterns

            hout = self.transferFunction(hin)
            if self.bias:
                hout =  np.concatenate((hout, np.ones((1, np.shape(x_train)[1]))), axis=0)

            oin = self.weights[1] @ hout
            out = self.transferFunction(oin)
            out_vec = [hout, out]
            return out_vec

        def backprop(out_vec, y_train):
            """
            Description:\n
            Backprop function\n
            Input:\n
                out_vec: the output vector for each layer\n
                y_train: target label\n

            Output:\n
                delta_h: the delta for the hidden layer\n
                delta_o: the delta for the output layer\n
            """
            #print(np.shape(out_vec[1]))
            delta_o = (out_vec[1] - y_train) * ((1 + out_vec[1]) * (1 - out_vec[1])) * 0.5
            delta_h = (np.transpose(self.weights[1]) @ delta_o) * ((1 + out_vec[0]) * (1 - out_vec[0])) * 0.5
            delta_h = delta_h[0:self.layers[0], :]
            return delta_h, delta_o

        # Inital delta weights.
        dw = np.zeros(np.shape(self.weights[0]))
        dv = np.zeros(np.shape(self.weights[1]))
        
        progBar = progressBar(epochs)
        loss_vec_train = []
        loss_vec_valid = []
        epoch_vec = []
        

        # training for all the epochs.
        for epoch in range(epochs):
            progBar.Progress(epoch)
            # Forwarding
            out_vec = forwardpass(x_train=x_train)

            # Back propogating
            delta_hidden, delta_output = backprop(out_vec, y_train)

            # Weights update
            if self.bias:
                pat = np.concatenate((x_train, np.ones((1, np.shape(x_train)[1]))))
            else:
                pat = x_train
            dw = (dw * alpha) - (delta_hidden @ np.transpose(pat)) * (1 - alpha)
            dv = (dv * alpha) - (delta_output @ np.transpose(out_vec[0])) * (1 - alpha)

            self.weights[0] = self.weights[0] + dw*eta
            self.weights[1] = self.weights[1] + dv*eta

            # Loss function 
            loss_vec_train.append(self.loss_val(x_train, target=y_train))
            loss_vec_valid.append(self.loss_val(x_valid, target=y_valid))
            epoch_vec.append(epoch)
        return epoch_vec, loss_vec_train, loss_vec_valid


def decision_boundary_multilayer(NeuralNet, x_train, y_train, x_train_A, x_train_B, disc=100, axis=[-1.5, 1.5]):
    x_val = np.linspace(axis[0], axis[1], disc).reshape(1,disc)
    y_val = np.linspace(axis[0], axis[1], disc).reshape(1,disc)
    
    x_vec = x_val
    for i in range(disc-1):
        x_vec = np.concatenate((x_vec,x_val), axis=1)
    XY = np.concatenate((x_vec, x_vec), axis=0)

    index = 0
    for y in y_val[0]:
        for i in range(disc):
            XY[1,index] = y
            index += 1


    Z = NeuralNet.forwardpass(XY)
    z = Z.reshape(disc,disc)
    x = XY[0].reshape(disc,disc)
    y = XY[1].reshape(disc,disc)

    plt.figure("Decision Boundary")
    plt.contourf(x, y, z, cmap = 'jet')
    plt.colorbar()
    NeuralNet.eval(x_train,y_train,verbose=True)
    plt.scatter(x_train_A[0,:], x_train_A[1,:], marker="+", c="black")
    plt.scatter(x_train_B[0,:], x_train_B[1,:], marker="_", c="black")

def main():
    n = 100
    bias = True
    x_train, y_train, x_valid, y_valid, x_train_A, x_train_B = generateClassData(n, proc_A=1, proc_B=1, case=2)

    NN = neuralNetwork(bias=bias, layers=[200, 1])
    NN.initWeights()
    epoch_vec, loss_vec_train, loss_vec_val = NN.train(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, epochs=1000, eta=0.1, alpha=0)


    plt.figure("Learning Curve")
    plt.plot(epoch_vec, loss_vec_train)
    plt.plot(epoch_vec, loss_vec_val)
    plt.legend(("Training loss", "Validation loss"))
    
    NN.eval(x_valid, y_valid, verbose=True)

    decision_boundary_multilayer(NeuralNet=NN,
                                 x_train=x_train, 
                                 y_train=y_train, 
                                 x_train_A=x_train_A, 
                                 x_train_B=x_train_B, 
                                 disc=100, 
                                 axis=[-1.5, 1.5])
    plt.show()
if __name__ == "__main__":
    main()