import numpy as np
import matplotlib.pyplot as plt

def generateClassData(n, proc_A, proc_B, case=2,verbose=False):
    
    if case==1:
        mA = np.array([1, 0.5])
        mB = np.array([-1, 0])
        sigmaA = 0.5
        sigmaB = 0.5

        # Take out the size of the classes
        classA_size = mA.size
        classB_size = mB.size

        # Create datapoints for A and B
        classA = np.random.randn(
            classA_size, n)*sigmaA + np.transpose(np.array([mA]*n))
        classB = np.random.randn(
            classB_size, n)*sigmaB + np.transpose(np.array([mB]*n))

        # Merges the datapoints from class A and class B and adding a target vector
        classData = np.concatenate((classA, classB), axis=1)
        T = np.concatenate((np.ones((1, n)), np.ones((1, n))*(-1)), axis=1)

        # Shuffler the data and the targets
        shuffler = np.random.permutation(n+n)
        classData = classData[:, shuffler]
        T = T[:, shuffler]

        if verbose:
            plt.scatter(classA[0,:], classA[1,:], c="red")
            plt.scatter(classB[0,:], classB[1,:], c="blue")
            plt.show()

        return classData, T
    elif case==2:
        mA = np.array([1.0, 0.3])
        mB = np.array([0.0, -0.1])
        sigmaA = 0.2
        sigmaB = 0.3

        # Creates class A data
        classA = np.random.randn(1, round(0.5*n)) * sigmaA - mA[0]
        classA = np.concatenate((classA, np.random.randn(1, round(0.5*n)) * sigmaA + mA[0]), axis=1)
        classA = np.concatenate((classA, np.random.randn(1, n) * sigmaA + mA[1]), axis=0)
    
        # Creates class B data
        classB = np.random.randn(1, n) * sigmaB + mB[0]
        classB = np.concatenate((classB, np.random.randn(1, n) * sigmaB + mB[1]), axis=0)

        # Creates target data
        T_A = np.ones((1, n))
        T_B = np.ones((1, n))*(-1)

        # Shuffles the data sets
        shuffler = np.random.permutation(n)
        classA = classA[:, shuffler]
        classB = classB[:, shuffler]
        T_A = T_A[:, shuffler]
        T_B = T_B[:, shuffler]

        # Take out the wanted procentage from each dataset
        x_train_A = classA[:,:round(n*proc_A)]
        x_train_B = classB[:,:round(n*proc_B)]
        y_train_A = T_A[:,:round(n*proc_A)]
        y_train_B = T_B[:,:round(n*proc_B)]
        
        # Create validation data
        x_valid_A = classA[:,round(n*proc_A):]
        x_valid_B = classB[:,round(n*proc_B):]

        y_valid_A = T_A[:,round(n*proc_A):]
        y_valid_B = T_B[:,round(n*proc_B):]

        x_valid = np.concatenate((x_valid_A, x_valid_B), axis=1)
        y_valid = np.concatenate((y_valid_A, y_valid_B), axis=1)

        # concatenates the datasets and the targets
        x_train = np.concatenate((x_train_A, x_train_B), axis=1)
        y_train = np.concatenate((y_train_A, y_train_B), axis=1)

        # Shuffles the dataset and the target
        shuffler = np.random.permutation(round(n*proc_A) + round(n*proc_B))
        x_train = x_train[:, shuffler]
        y_train = y_train[:, shuffler]
        
        if verbose:
            plt.figure("Genererated Data")
            plt.scatter(x_train_A[0,:], x_train_A[1,:], c="red")
            plt.scatter(x_train_B[0,:], x_train_B[1,:], c="blue")
            plt.scatter(x_valid_A[0,:], x_valid_A[1,:], c="red")
            plt.scatter(x_valid_B[0,:], x_valid_B[1,:], c="blue")

        return x_train, y_train, x_valid, y_valid, x_train_A, x_train_B
    elif case==3:
        mA = np.array([1.0, 0.3])
        mB = np.array([0.0, -0.1])
        sigmaA = 0.2
        sigmaB = 0.3

        # Creates class A data
        classA = np.random.randn(1, round(0.4*n)) * sigmaA - mA[0]
        classA = np.concatenate((classA, np.random.randn(1, round(0.1*n)) * sigmaA + mA[0]), axis=1)
        classA = np.concatenate((classA, np.random.randn(1, round(0.4*n + 0.1*n)) * sigmaA + mA[1]), axis=0)

        valid_A = np.random.randn(1, round(0.1*n)) * sigmaA - mA[0]
        valid_A = np.concatenate((valid_A, np.random.randn(1, round(0.4*n)) * sigmaA + mA[0]), axis=1)
        valid_A = np.concatenate((valid_A, np.random.randn(1, round(0.4*n + 0.1*n)) * sigmaA + mA[1]), axis=0)

        # Creates class B data
        classB = np.random.randn(1, n) * sigmaB + mB[0]
        classB = np.concatenate((classB, np.random.randn(1, n) * sigmaB + mB[1]), axis=0)

        # Creates target data
        T_A = np.ones((1,round(0.4*n + 0.1*n)))
        T_B = np.ones((1, n))*(-1)


        # Take out the wanted procentage from each dataset
        x_train_A = classA
        x_train_B = classB[:,:round(n*proc_B)]
        y_train_A = T_A[:,:round(n*proc_A)]
        y_train_B = T_B[:,:round(n*proc_B)]
        
        # Create validation data
        x_valid_A = valid_A
        x_valid_B = classB[:,round(n*proc_B):]

        y_valid_A = np.ones((1,round(0.1*n + 0.4*n)))
        y_valid_B = T_B[:,round(n*proc_B):]

        x_valid = np.concatenate((x_valid_A, x_valid_B), axis=1)
        y_valid = np.concatenate((y_valid_A, y_valid_B), axis=1)

        # concatenates the datasets and the targets
        x_train = np.concatenate((x_train_A, x_train_B), axis=1)
        y_train = np.concatenate((y_train_A, y_train_B), axis=1)

        # Shuffles the dataset and the target
        shuffler = np.random.permutation(np.shape(x_train)[1])
        x_train = x_train[:, shuffler]
        y_train = y_train[:, shuffler]
        
        if verbose:
            plt.figure("Genererated Data")
            plt.scatter(x_train_A[0,:], x_train_A[1,:], c="red")
            plt.scatter(x_train_B[0,:], x_train_B[1,:], c="blue")
            plt.scatter(x_valid_A[0,:], x_valid_A[1,:], c="red")
            plt.scatter(x_valid_B[0,:], x_valid_B[1,:], c="blue")

        return x_train, y_train, x_valid, y_valid, x_train_A, x_train_B