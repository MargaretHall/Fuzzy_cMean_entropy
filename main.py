import random

import pandas
import sklearn
from matplotlib import pyplot as plt
import run_on
import numpy as np

if __name__ == '__main__':

    # run_on.run_on_digits()

    c = 15 # how many times to run fuzzy c-means

    # shows entropy of several clustering attempts (with 3 clusters)
    stability = []
    average = 0
    min_entropy = 500
    max_entropy = 0
    min_memb = []
    max_memb = []
    x = []
    y = []
    seed = [0.6027417733363224, 0.7627736740149785, 0.5066953208709329, 0.3546213390689169, 0.7735797645463781,
            0.12940520781022746, 0.5179667903182829, 0.7778286039773062, 0.5102474499180927, 0.0032474887350526505,
            0.5860486090543451, 0.9269833407471108, 0.06595287935447192, 0.9472413877337412, 0.296626764242371]
    for i in range(c):
        #seed = random.random()
        #print(seed)

        # set show_graphs = True on run_iris to view clusters and entropy graphs of every iteration
        # run without passing an argument for seed for random clusters. Seed list will produce results shown in paper
        entropy, memb, x, y = run_on.run_iris(seed[i], show_graphs=False)
        stability.append(entropy)
        if entropy < min_entropy:
            min_entropy = entropy
            min_memb = memb
        if entropy > max_entropy:
            max_entropy = entropy
            max_memb = memb
        #average += entropy
    #average = average / c


    plt.scatter(range(1, c + 1), stability)
    plt.ylabel("Total Entropy")
    plt.show()

    plt.scatter(x, y, c=min_memb)
    plt.ylabel("PC2")
    plt.xlabel("PC1")
    plt.show()
    print("Minimum entropy iteration, entropy= ", min_entropy)

    plt.scatter(x, y, c=max_memb)
    plt.ylabel("PC2")
    plt.xlabel("PC1")
    plt.show()
    print("Maximum entropy iteration, entropy= ", max_entropy)

    true_clust = np.array(pandas.read_csv('iris_true_clusters2.csv')).T[0]

    num_round = 1
    min_memb = min_memb.round(num_round).T[0]
    max_memb = max_memb.round(num_round).T[0]

    MI = sklearn.metrics.mutual_info_score(true_clust, min_memb)
    MI2 = sklearn.metrics.mutual_info_score(true_clust, max_memb)
    print("Lowest entropy run, mutual information= ", MI)
    print("Highest entropy run, mutual information= ",MI2)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
