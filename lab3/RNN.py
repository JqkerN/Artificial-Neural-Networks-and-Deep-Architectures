import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits import mplot3d
import itertools
import time
# from alive_progress import alive_bar


class RNN():
    def __init__(self, size=8, diagonal=True, sequential=False, random=False):
        self.size = size
        self.diagonal = diagonal
        self.weights = np.zeros((size,size))
        self.sequential=sequential
        self.random = random
    
    def init_weights(self, patterns):
        patterns_copy=patterns.copy()

        if self.sequential:
            for i in range(self.size):
                for j in range(self.size): 
                    for pattern in patterns_copy: 
                        self.weights[i][j] = self.weights[i][j] + pattern[0][i]*pattern[0][j]
            self.weights /= self.size

        elif self.diagonal:
            for x in patterns:
                self.weights += (np.transpose(x) @ x)
            self.weights /= self.size

        else:
            for x in patterns:
                self.weights += (np.transpose(x) @ x)
            self.weights /= self.size
            np.fill_diagonal(self.weights, 0)

    def update(self, x):
        if self.sequential and self.random:
            random_vec = np.random.permutation(np.linspace(0, self.size - 1, self.size))
            x_new = x.copy()
            for i in random_vec:
                i = int(i)
                sum_tmp=0
                for j in random_vec:
                    j = int(j)
                    sum_tmp += self.weights[i][j]*x_new[0][j]
                if sum_tmp < 0:
                    x_new[0][i] = -1
                else: 
                    x_new[0][i] = 1
            x_new = np.transpose(x_new)

        elif self.sequential:
            x_new = x.copy()
            for i in range(self.size):
                sum_tmp=0
                for j in range(self.size):
                    sum_tmp += self.weights[i][j]*x_new[0][j]
                if sum_tmp < 0:
                    x_new[0][i] = -1
                else: 
                    x_new[0][i] = 1
            x_new = np.transpose(x_new)

        else:    
            tmp = self.weights @ np.transpose(x)
            x_new = np.where(tmp>=0, 1, -1)
        return np.transpose(x_new)

    def train(self, x):
        x_old = np.zeros((1,self.size)) 
        x_new = x  
        iteration = 0

        while not (x_new == x_old).all() and iteration < 1000:
            iteration += 1
            x_old = x_new
            x_new = self.update(x_old)

            # print("Iteration: {}".format(iteration))
        return x_new


    def layapunovFunction(self, x, verbose=True):
        energi = 0
        for i in range(self.size):
            for j in range(self.size):
                energi -= self.weights[i][j]*x[0][i]*x[0][j]
        # if verbose:
        #     fig = plt.figure()
        #     ax = plt.axes(projection='3d')
        #     X, Y = np.meshgrid(x, x)
        #     ax.plot_surface(X, Y, energi, rstride=1, cstride=1,
        #             cmap='jet', edgecolor='none')

        #     ax.set_title('finaplot');
        return energi






def main():
    # Original patterns
    x1 = np.array([[-1, -1, 1, -1, 1, -1, -1 ,1]])
    x2 = np.array([[-1, -1, -1, -1, -1, 1, -1 ,-1]])
    x3 = np.array([[-1, 1, 1, -1, -1, 1, -1 ,1]])
    print(x1.shape)
    # Distorted patterns
    x1d = np.array([[1, -1, 1, -1, 1, -1, -1 ,1]])
    x2d = np.array([[1, 1, -1, -1, -1, 1, -1 ,-1]])
    x3d = np.array([[1, 1, 1, -1, 1, 1, -1 ,1]])
    
    patterns = [x1, x2, x3]
    patterns_distorted = [x1d, x2d, x3d]
    network = RNN(sequential=False)                      # SEQuAENTIOAL WARNING
    network.init_weights(patterns)
    print(patterns)

    for xd, x in zip(patterns_distorted, patterns):
        x_output = network.train(xd)
        print('Number of correct: {}/{} '.format(np.count_nonzero(x_output==x), x.shape[1]))


    print('\nMore than half distorted: ')
    x4d = np.array([[1, -1, 1, -1, 1, 1, -1 ,-1]])
    x_output = network.train(x4d)
    print('Number of correct: {}/{} '.format(np.count_nonzero(x_output==x1), x1.shape[1]))
    print('Number of correct: {}/{} '.format(np.count_nonzero(x_output==x2), x2.shape[1]))
    print('Number of correct: {}/{} '.format(np.count_nonzero(x_output==x3), x3.shape[1]))
    print('Output: ', x_output)


    # Taking out all possible attractors
    lst = list(itertools.product([-1, 1], repeat=8))
    attractors = []
    for xd in lst:
        xd = np.array([xd])
        x_output = network.train(xd)
        x_output = np.ndarray.tolist(x_output)
        if not np.all(x_output in attractors):
            attractors.append(x_output)
    
    print('\nNumber of unique attractors: ',len(attractors))
















if __name__ == '__main__':
    main()
