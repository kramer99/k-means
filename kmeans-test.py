import numpy as np
import matplotlib.pyplot as plt
import kmeans

def twoD():
    m = 100  # sample size
    n = 2   # number of features
    K = 3   # number of clusters
    X = np.random.uniform(0, 100, (m,n))
    clusterings = kmeans.run(X, K)

def threeD():
    m = 200  # sample size
    n = 3   # number of features
    K = 4   # number of clusters
    X = np.random.uniform(0, 100, (m,n))
    clusterings = kmeans.run(X, K)

twoD()