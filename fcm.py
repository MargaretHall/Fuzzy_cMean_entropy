import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


# parameters: data matrix, k the initialized clusters, and m the "fuzzifier"
def fuzzy_c_means(data_matrix, k, m=2.0):
    reps = 15
    epsilon = .3

    # this init is problematic I think. instead randomly assign cluster centers to start and then calculate membership
    #init = 1 / k  # start with it being equally likely a point is in each cluster
    # Uij = np.array([[init for point in range(k)] for cluster in
    #                 range(len(data_matrix))], dtype=float)  # membership to cluster j for data point i

    #init_centers = np.array([[5, 3.5, 1.5, .2], [6.5, 3, 4.5, 1.2], [7, 3, 5.5, 1.5]])

    init_centers = k

    memb_k = calc_memb(data_matrix, init_centers, m)
    # print(Uij)
    # print(len(data_matrix))
    norm = epsilon
    entropy_progression_total = list()
    entropy_progression_average = list()
    while norm >= epsilon:
        new_center = cluster_center(data_matrix, memb_k, m)
        entropy = calc_entropy(memb_k)
        total_entropy = sum(entropy)
        average_entropy = np.mean(entropy)
        entropy_progression_total.append(total_entropy)
        entropy_progression_average.append(average_entropy)

        #print(total_entropy, ", ", average_entropy)

        '''plots each iteration of clustering'''
        #plotit(memb_k, data_matrix)

        memb_k1 = calc_memb(data_matrix, new_center, m)
        norm = np.linalg.norm(memb_k - memb_k1)
        #if norm < epsilon:
            #print("norm (",  norm, ") < epsilon")
        memb_k = memb_k1

    #lowest entropy: 85.79928194760535

    return memb_k, entropy_progression_average, entropy_progression_total


# parameters: data is the data matrix being clustered,
# membership is a matrix containing the membership of each data point to each cluster
# m is the same "fuzzifier"
def cluster_center(data, membership, m):
    membXfuzz = (np.power(membership, m)).T  #where each row represents the relation of the data point to each cluster,
                                             #and each column is a new data point
    centers = list()

    for cluster in membXfuzz:
        data2 = data.copy()
        for i in range(len(cluster)):
            # print(i)
            # print("pre: \n", data[i])
            data2[i] = data2[i] * cluster[i]
            # print("post: \n", data2[i])
        # print("sum", sum(cluster))
        centers.append(np.sum(data2, axis=0) / sum(cluster))
        # print(centers)

    centers = np.array(centers, dtype=float)
    return centers


# in centers, each row represents a different cluster, and each column represents the value in dimension
def calc_memb(data, centers, m):
    power = 2 / (m - 1)
    edm = euclidean_matrix(data, centers)
    memb_array = np.empty_like(edm)

    # summation, for each cluster(k):
    # euclidean distance from point i to
    # cluster j/distance of point i from cluster k
    for j in range(len(centers)):
        for point in range(len(data)):
            # can loop through each cluster j, for each point i
            summation = 0
            memb = 0
            for cluster in range(len(centers)):
                #creates the summation
                if edm[point][cluster] == 0:
                    edm[point][cluster] = .000001 #since digit clusters are initialized as a digit, avoid divide by 0
                memb = edm[point][j] / edm[point][cluster]
                summation += math.pow(memb, power)
            memb_array[point][j] = 1 / summation

            # needs to create a matrix/array defining each point to each cluster

    return memb_array


def euclidean_matrix(data, clusters):
    # print(type(data))
    # print(type(clusters))
    # sqrt((uTu – 2uTv + vTv))
    verticle_norms_squared = np.sum(data ** 2, axis=1)[:, np.newaxis]
    horizontal_norms_squared = np.sum(clusters ** 2, axis=1)
    XXT2 = -2 * np.dot(data, clusters.T)
    M = verticle_norms_squared + XXT2 + horizontal_norms_squared
    M = np.array(M, dtype=float)  # idk it just works
    M = np.round(M, decimals=7)  # remove negative 0
    M = np.sqrt(M)

    return M

def calc_entropy(membership):
    log_array = np.log2(membership) #uses log base 2... this puts the "information" into bits
    mult = log_array * membership
    entropy = -np.sum(mult, axis=1)
    return entropy


def cluster_entropy(membership):
    for point in membership:
        print(point)
        #find largest
        #assign point to "most likely" cluster
        #calc each cluster entropy
        #return cluster entropies


def plotit(memb, iris):
    standard_iris = StandardScaler().fit_transform(iris)
    iris_pca = PCA(n_components=10)
    iris_pc = iris_pca.fit_transform(standard_iris)
    x = iris_pc[:, 0]
    y = iris_pc[:, 1]
    plt.scatter(x, y, c = memb)
    plt.show()
