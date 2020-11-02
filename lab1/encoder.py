import matplotlib.pyplot as plt
import numpy as np
import math
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

    def activationFunction(self, dataset):
        for index, data in enumerate(dataset):
            dataset[index][ data > 0] = 1
            dataset[index][ data <= 0] = -1
        return dataset

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
        return out, hout

    def classify(self, data):
        data, hout = self.forwardpass(data=data)
        output = self.activationFunction(data)
        print(output)
        return output, hout

    def eval(self, data, targets, verbose=False):
        classified_data, hout = self.classify(data)
        accuracy = np.count_nonzero(classified_data == targets)/np.shape(targets)[1]*100
        print("Accuracy: ", accuracy)
        if verbose:
            plt.figure("Decision Boundary")
            plt.scatter(data[0, np.where(classified_data==targets)], data[1, np.where(classified_data==targets)], c="green")
            plt.scatter(data[0, np.where(classified_data!=targets)], data[1, np.where(classified_data!=targets)], c="red")
        return hout  

    def loss_val(self, data, target):
        output, _ = self.forwardpass(data=data)
        loss = 1 / (2*np.shape(target)[1]) * np.sum( np.power(output - target, 2))
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
        
        progBar = progressBar(epochs)
        loss_vec_train = []
        loss_vec_valid = []
        epoch_vec = []
        

        # training for all the epochs.
        for epoch in range(epochs):

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
    
def generateClassData():
    data = np.transpose(np.array([[1,-1,-1,-1,-1,-1,-1,-1],[-1,-1,1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,1,-1,-1,-1],[-1,-1,-1,-1,-1,-1,1,-1],[-1,-1,-1,-1,-1,-1,-1,1],[-1,-1,-1,-1,-1,1,-1,-1],[-1,-1,-1,1,-1,-1,-1,-1],[-1,1,-1,-1,-1,-1,-1,-1]]))
    return data

def main():

    x_train = generateClassData()
    print(x_train)
    NN = neuralNetwork(layers=[3, 8])
    NN.initWeights(dim=8)
    epoch_vec, loss_vec_train = NN.train(x_train=x_train, y_train=x_train, epochs=100000, eta=0.001, alpha=0.5)
    
    plt.figure("Learning Curve")
    plt.plot(epoch_vec, loss_vec_train)
    plt.legend(("Training loss", "Validation loss"))
    plt.show()


    NN.eval(x_train, x_train, verbose=False)

    hout_1 = NN.eval(np.array(np.transpose([x_train[:,0]])), np.array(np.transpose([x_train[:,0]])), verbose=False)
    hout_2 = NN.eval(np.array(np.transpose([x_train[:,7]])), np.array(np.transpose([x_train[:,7]])), verbose=False)
    hout_3 = NN.eval(np.array(np.transpose([x_train[:,1]])), np.array(np.transpose([x_train[:,1]])), verbose=False)
    hout_4 = NN.eval(np.array(np.transpose([x_train[:,6]])), np.array(np.transpose([x_train[:,6]])), verbose=False)
    hout_5 = NN.eval(np.array(np.transpose([x_train[:,2]])), np.array(np.transpose([x_train[:,2]])), verbose=False)
    hout_6 = NN.eval(np.array(np.transpose([x_train[:,5]])), np.array(np.transpose([x_train[:,5]])), verbose=False)
    hout_7 = NN.eval(np.array(np.transpose([x_train[:,3]])), np.array(np.transpose([x_train[:,3]])), verbose=False)
    hout_8 = NN.eval(np.array(np.transpose([x_train[:,4]])), np.array(np.transpose([x_train[:,4]])), verbose=False)

    hout_1 = [float(hout_1[0]), float(hout_1[1]), float(hout_1[2])]
    hout_2 = [float(hout_2[0]), float(hout_2[1]), float(hout_2[2])]
    hout_3 = [float(hout_3[0]), float(hout_3[1]), float(hout_3[2])]
    hout_4 = [float(hout_4[0]), float(hout_4[1]), float(hout_4[2])]
    hout_5 = [float(hout_5[0]), float(hout_5[1]), float(hout_5[2])]
    hout_6 = [float(hout_6[0]), float(hout_6[1]), float(hout_6[2])]
    hout_7 = [float(hout_7[0]), float(hout_7[1]), float(hout_7[2])]
    hout_8 = [float(hout_8[0]), float(hout_8[1]), float(hout_8[2])]

    x = [1,2,3]
    axis = [0, 4, -2, 2]
    plt.figure("Encoder hidden layers acitivity")
    plt.subplot(4,2,1)
    plt.title("Pattern: 1")
    plt.bar(x, hout_1)
    plt.axis(axis)
    plt.subplot(4,2,2)
    plt.title("Pattern: 2")
    plt.bar(x, hout_2)
    plt.axis(axis)
    plt.subplot(4,2,3)
    plt.title("Pattern: 3")
    plt.bar(x, hout_3)
    plt.axis(axis)
    plt.subplot(4,2,4)
    plt.title("Pattern: 4")
    plt.bar(x, hout_4)
    plt.axis(axis)
    plt.subplot(4,2,5)
    plt.title("Pattern: 5")
    plt.bar(x, hout_5)
    plt.axis(axis)
    plt.subplot(4,2,6)
    plt.title("Pattern: 6")
    plt.bar(x, hout_6)
    plt.axis(axis)
    plt.subplot(4,2,7)
    plt.title("Pattern: 7")
    plt.bar(x, hout_7)
    plt.axis(axis)
    plt.subplot(4,2,8)
    plt.title("Pattern: 8")
    plt.bar(x, hout_8)
    plt.axis(axis)
    plt.show()
if __name__ == "__main__":
    main()