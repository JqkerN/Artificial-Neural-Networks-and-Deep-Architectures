from RNN import RNN
import numpy as np
import math
from mpl_toolkits import mplot3d
import itertools
import matplotlib.pyplot as plt


def loadData(file):
    inputData = np.array([])
    with open(file, 'r') as f:
        d = f.readline()
        k = d.split(",")
        results = list(map(int, k))
        data = np.array(results).reshape(11, 1024)
    return data


    




def main():

    patterns = loadData('pict.dat')     # Pattern 1-11
    patterns_1_3 = [patterns[index,:].reshape(1,1024) for index in range(3) ]
    patterns_4_11 = [patterns[3+index,:].reshape(1,1024) for index in range(8) ]
    

    network = RNN(size=1024, sequential=False, random=True)
    network.init_weights(patterns_1_3)


    for index, pattern in enumerate(patterns_1_3):
        energi = network.layapunovFunction(pattern)
        print('Energi for pattern {}: {}'.format(index, energi))

    for index, pattern in enumerate(patterns_4_11):
        energi = network.layapunovFunction(pattern)
        print('Energi for distorted pattern {}: {}'.format(3 + index, energi))



if __name__ == "__main__":
    main()

