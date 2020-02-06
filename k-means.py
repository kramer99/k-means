import numpy as np
import matplotlib.pyplot as plt

def run():
    m = 100  # sample size
    n = 2   # number of features
    K = 3   # number of clusters
    X = np.random.uniform(0, 100, (m,n))

    # pick K initial centroids from positions of randomly selected samples
    indexes = np.random.randint(0, m, K)
    centroids = X.take(indexes, axis = 0)

    converged = False
    J_history = []

    while not converged:
        clusterings = assignClusters(X, centroids)
        centroids, J = moveCentroids(centroids, X, clusterings)
        print('cost:',J)
        J_history.append(J)
        if len(J_history) > 1 and J >= J_history[-2]:   # keep going until cost stops decreasing
            converged = True
        plotClusters(X, clusterings, centroids)

    plt.plot(np.arange(0,len(J_history)), J_history, label='J')

def assignClusters(X, centroids):
    m = len(X)
    clusterings = np.zeros(m)
    for i in range(m):
        distances = []
        for mu in centroids:
            distances.append(np.linalg.norm(X[i] - mu))
        clusterings[i] = np.argmin(distances)
    return clusterings

def moveCentroids(centroids, X, clusterings):
    newCentroids = []
    J = 0
    for k in range(len(centroids)):
        # we take the average position in feature space of all samples assigned
        # to centroid k, and move that centroid to this new average position
        indexes = np.where(clusterings == k)[0]
        cluster = X.take(indexes, axis = 0)
        cluster = cluster.reshape(len(indexes), X.shape[1]) # np.take adds an extra dimension for some reason
        mu = np.mean(cluster, axis = 0)
        newCentroids.append(mu)

        # compute the cost here while we have the data at hand
        J = J + np.sum(np.square(np.linalg.norm(cluster - mu, axis=1)))
    J = J / len(X)
    return newCentroids, J

def plotClusters(X, clusterings, centroids):
    K = len(centroids)
    for k in range(K):
        indexes = np.where(clusterings == k)[0]
        cluster = X.take(indexes, axis = 0)
        plt.scatter(cluster[:,0], cluster[:,1])
        plt.plot(centroids[k][0], centroids[k][1], '+')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

run()