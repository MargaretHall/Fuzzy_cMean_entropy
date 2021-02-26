import random
import numpy as np
import pandas
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import fcm
from matplotlib import pyplot as plt


def run_iris(seed=None, show_graphs = False):
    iris = pandas.read_csv('iris_csv.csv')
    iris = iris.to_numpy()
    iris = (iris[0:, 0:4])  # data set without cluster name as matrix, curr

    if seed == None:
        random.seed()
    else:
        random.seed(seed)
    init = np.array([[random.randrange(4, 8), random.randrange(2, 4), random.randrange(1, 6), random.randrange(5, 8)],
                     [random.randrange(4, 8), random.randrange(2, 4), random.randrange(1, 6), random.randrange(5, 8)],
                     [random.randrange(4, 8), random.randrange(2, 4), random.randrange(1, 6), random.randrange(5, 8)]])
    memb_mat, avg_entropy, tot_entropy = fcm.fuzzy_c_means(iris, init)

    memb_mat1 = memb_mat[:, 0:1] * 1
    memb_mat2 = memb_mat[:, 1:2] * 2
    memb_mat3 = memb_mat[:, 2:] * 3
    weight_mm = np.sum((memb_mat1, memb_mat2, memb_mat3), axis=0)
    # weight_mm2 = np.concatenate((memb_mat1, memb_mat2, memb_mat3), axis=1)

    standard_iris = StandardScaler().fit_transform(iris)
    iris_pca = PCA(n_components=2)
    iris_pc = iris_pca.fit_transform(standard_iris)
    x = iris_pc[:, :1]
    y = iris_pc[:, 1:]

    if show_graphs:
        plt.scatter(x, y, c=weight_mm)
        plt.show()

        plt.plot(avg_entropy)
        plt.ylabel("Average Entropy")
        plt.xlabel("k")
        plt.show()

        plt.plot(tot_entropy)
        plt.ylabel("Total Entropy")
        plt.xlabel("k")
        plt.show()

    return(tot_entropy[-1], weight_mm, x, y)

'''Ultimetaly unused method to run iris clustering with variable number of clusters, x'''
def run_x_clust(x):
    iris = pandas.read_csv('iris_csv.csv')
    iris = iris.to_numpy()
    iris = (iris[0:, 0:4])  # data set without cluster name as matrix, curr

    #create centers for each cluster x
    centers = []
    for i in range(x):
        centers.append([random.randrange(4, 8), random.randrange(2, 4), random.randrange(1, 6), random.randrange(5, 8)])
    centers = np.array(centers)

    memb_mat, avg_entropy, tot_entropy = fcm.fuzzy_c_means(iris, centers)

    #colorize clusters
    mm = []
    for i in range(x):
        #mult = memb_mat[:, i:i+1]
        #multt = mult * (i+1)
        mm.append(memb_mat[:, i:i + 1] * (i + 1))
    mm = np.array(mm)
    mm = np.reshape(mm, (150, i + 1))
    weight_mm = np.sum(mm, axis=1)
    print(np.shape(weight_mm))

    standard_iris = StandardScaler().fit_transform(iris)
    iris_pca = PCA(n_components=2)
    iris_pc = iris_pca.fit_transform(standard_iris)
    x = iris_pc[:, :1]
    y = iris_pc[:, 1:]

    '''plt.scatter(x, y, c=weight_mm)
    plt.show()

    plt.plot(avg_entropy)
    plt.show()

    plt.plot(tot_entropy)
    plt.show()'''

    return tot_entropy[-1]

'''incomplete function to cluster digits data set'''
def run_on_digits():
    digits = pandas.read_csv('digits.csv')
    digits = digits.to_numpy()

    standard_digit = StandardScaler().fit_transform(digits)
    digit_pca = PCA(n_components=10)
    digit_pc = digit_pca.fit_transform(standard_digit)
    x = digit_pc[:, 0]
    y = digit_pc[:, 1]

    init = np.array((digit_pc[random.randrange(0, 1795)], digit_pc[random.randrange(0, 1795)], digit_pc[random.randrange(0, 1795)],
                    digit_pc[random.randrange(0, 1795)], digit_pc[random.randrange(0, 1795)], digit_pc[random.randrange(0, 1795)],
                    digit_pc[random.randrange(0, 1795)], digit_pc[random.randrange(0, 1795)], digit_pc[random.randrange(0, 1795)],
                    digit_pc[random.randrange(0, 1795)]))
    memb_mat, avg_entropy, tot_entropy = fcm.fuzzy_c_means(digit_pc, init)

    mm = []
    for i in range(10):
        #mult = memb_mat[:, i:i+1]
        #multt = mult * (i+1)
        mm.append(memb_mat[:, i:i + 1] * (i + 1))
    mm = np.array(mm)
    mm.reshape(1796,10)
    #mm = np.reshape(mm, (1796, i + 1))
    weight_mm = np.sum(mm, axis=0)
    print(np.shape(weight_mm))

    plt.scatter(x, y, c=weight_mm)
    plt.show()

    print("done")