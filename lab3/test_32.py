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
    patterns_10_11 = [patterns[9+index,:].reshape(1,1024) for index in range(2) ]

    network = RNN(size=1024, sequential=True, random=True)
    network.init_weights(patterns_1_3)




    # Testing if stable
    print('\nTesting if stable: ')
    plt.figure('Attractors - patterns')
    index = 0
    for pattern in patterns_1_3:
        index += 1
        plt.subplot(1, 3, index)
        plt.title('Pattern: {}'.format(index))
        x_output = network.train(pattern)
        print('Number of correct: {}/{} '.format(np.count_nonzero(x_output==pattern), pattern.shape[1]))
        plt.imshow(pattern.reshape(32,32), cmap='gray')





    # Testing for distorted patterns 9 and 10
    print('\nTesting for distorted patterns: ')
    index = 10
    for pattern in patterns_10_11:
        plt.figure('Output - pattern: {}'.format(index))
        index += 1
        sub_index = 0 
        x_output = network.train(pattern)
        for true_pattern in patterns_1_3:
            sub_index += 1
            print('Number of correct: {}/{} '.format(np.count_nonzero(x_output==true_pattern), true_pattern.shape[1]))
        plt.imshow(x_output.reshape(32,32), cmap='gray')
    plt.show()




if __name__ == "__main__":
    main()









# def readData(file):
#     inputData = np.array([])
#     with open(file, 'r') as f:
#         d = f.readlines()
#         index=0
#         for i in d:
#             k = i.rstrip().split("\t")
#             k=([i.split(" ") for i in k])
#             if index==0:
#                 inputData = np.array([float(i) for i in k[0]])
#                 outputData = np.array([float(i) for i in k[1]])
#             else:
#                 inputData=np.vstack((inputData,([float(i) for i in k[0]])))
#                 outputData=np.vstack((outputData,([float(i) for i in k[1]])))
#             index=1
#     return inputData, outputData