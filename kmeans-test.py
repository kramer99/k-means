import numpy as np
import matplotlib.pyplot as plt
import kmeans

def uniformTwoD():
    m = 100  # sample size
    n = 2   # number of features
    K = 3   # number of clusters
    X = np.random.uniform(0, 100, (m,n))
    clusterings = kmeans.run(X, K)

def uniformThreeD():
    m = 200  # sample size
    n = 3   # number of features
    K = 4   # number of clusters
    X = np.random.uniform(0, 100, (m,n))
    clusterings = kmeans.run(X, K)

def clusteredTwoD():
    m = 100  # sample size
    n = 2   # number of features
    K = 3   # number of clusters
    X = np.zeros((0,n))
    centersOfMass = np.random.uniform(0, 100, (K,n))
    for i in centersOfMass:
        stdDev = 12
        samples = np.random.normal(i, stdDev, (int(m / K), n))
        X = np.append(X, samples, axis = 0)
    clusterings = kmeans.run(X, K)

def clusteredThreeD():
    m = 200  # sample size
    n = 3   # number of features
    K = 4   # number of clusters
    X = np.zeros((0,n))
    centersOfMass = np.random.uniform(0, 100, (K,n))
    for i in centersOfMass:
        stdDev = 12
        samples = np.random.normal(i, stdDev, (int(m / K), n))
        X = np.append(X, samples, axis = 0)
    clusterings = kmeans.run(X, K)

clusteredTwoD()