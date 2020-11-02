import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from generateData import generateClassData
from progressBar import progressBar
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from alive_progress import alive_bar

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

    def train(self, x_train, y_train, epochs, eta=0.001, alpha=0.9):
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
        
        # progBar = progressBar(epochs)
        
        loss_vec_train = []
        epoch_vec = []
        
        with alive_bar(epochs) as bar:
            # training for all the epochs.
            for epoch in range(epochs):
                bar()
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
                epoch_vec.append(epoch)
            return epoch_vec, loss_vec_train


def decision_boundary_multilayer(NeuralNet, x_train, y_train, disc=100, axis=[-1.5, 1.5]):
    x_val = np.linspace(axis[0], axis[1], disc).reshape(1,disc)
    y_val = np.linspace(axis[0], axis[1], disc).reshape(1,disc)
    
    x_vec = x_val
    for _ in range(disc-1):
        x_vec = np.concatenate((x_vec,x_val), axis=1)
    XY = np.concatenate((x_vec, x_vec), axis=0)

    index = 0
    for y in y_val[0]:
        for _ in range(disc):
            XY[1,index] = y
            index += 1


    Z = NeuralNet.forwardpass(XY)
    z = Z.reshape(disc,disc)
    x = XY[0].reshape(disc,disc)
    y = XY[1].reshape(disc,disc)

    plt.figure("Decision Boundary")
    plt.contourf(x, y, z, cmap = 'jet')
    plt.colorbar()

def bellcurve(NeuralNet, x_train, axis=[-5,5], delta=0.5):
    X = np.arange(-5, 5+delta, delta)
    Y = np.arange(-5, 5+delta, delta)
    X, Y = np.meshgrid(X, Y)
    Z = NeuralNet.forwardpass(x_train)

    Z = Z.reshape(int((axis[1]-axis[0])/delta + 1), int((axis[1]-axis[0])/delta + 1))

    fig = plt.figure("3D Bellcurve function")
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap="jet",
                       linewidth=0, antialiased=False)

    # # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()




def generateBelcurveData(axis=[-5,5], delta=0.5):
    x_val = np.arange(-5, 5+delta, delta).reshape(1,-1)
    y_val = np.arange(-5, 5+delta, delta).reshape(1,-1)

    x_vec = x_val
    for _ in range(x_val.shape[1]-1):
        x_vec = np.concatenate((x_vec,x_val), axis=1)
    x_train = np.concatenate((x_vec, x_vec), axis=0)

    index = 0
    for y in y_val[0]:
        for _ in range(y_val.shape[1]):
            x_train[1,index] = y
            index += 1

    y_train = np.transpose(np.exp(-x_val*x_val*0.1)) @ np.exp(-y_val*y_val*0.1) - 0.5
    y_train = y_train.reshape(1,-1)
    return x_train, y_train

def main():
    bias = True
    delta = 0.5
    datasplit = 0.7
    
    x_valid, y_valid = generateBelcurveData(delta=delta)

    n = np.size(y_valid, 1)
    shuffler = np.random.permutation(n)
    x_train = x_valid[:, shuffler]
    y_train = y_valid[:, shuffler]
    x_train = x_train[:,:round(n*datasplit)]
    y_train = y_train[:,:round(n*datasplit)]

    NN = neuralNetwork(bias=bias, layers=[4, 1])
    NN.initWeights()
    epoch_vec, loss_vec_train = NN.train(x_train=x_train, y_train=y_train, epochs=1000, eta=0.01, alpha=0.9)


    plt.figure("Learning Curve")
    plt.plot(epoch_vec, loss_vec_train)
    plt.legend(("Training loss", "Validation loss"))
    
    print("MSE Value of validation data: {:.14f}".format(NN.loss_val(x_valid, y_valid)))

    decision_boundary_multilayer(NeuralNet=NN,
                                 x_train=x_valid, 
                                 y_train=y_valid, 
                                 disc=100, 
                                 axis=[-6, 6])
    
    bellcurve(NN, x_valid, delta=delta)

if __name__ == "__main__":
    main()