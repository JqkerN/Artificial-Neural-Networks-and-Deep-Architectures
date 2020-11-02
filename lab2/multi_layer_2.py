import matplotlib.pyplot as plt
import numpy as np
import math
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

    def initWeights(self, dim=1, sigma=1):
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
        return data

          
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
        

        loss_vec_train = []
        loss_vec_valid = []
        epoch_vec = []

        old_val_loss = 100000000
        val_loss = 10000000
        epoch = 1
        with alive_bar(epochs) as bar:
            # training for all the epochs.
            while epoch < 50000:
                # if epoch > 2000000:
                #     break
            # for epoch in range(epochs):
                bar()
                old_val_loss = val_loss
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
                val_loss = self.loss_val(x_valid, target=y_valid)
                loss_vec_train.append(self.loss_val(x_train, target=y_train))
                loss_vec_valid.append(val_loss)
                epoch_vec.append(epoch)
                epoch += 1
                #print( "Validation loss: {}".format(val_loss), end='\r', flush = True)
        return epoch_vec, loss_vec_train, loss_vec_valid
