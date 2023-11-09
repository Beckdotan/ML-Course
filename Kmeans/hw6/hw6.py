import numpy as np
import matplotlib.pyplot as plt

def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Ensure that k is not greater than the number of pixels in the image
    k = min(k, X.shape[0])

    # Randomly choose 'k' indices from the range of data points
    random_indices = np.random.choice(X.shape[0], size=k, replace=False)

    # Select the data points corresponding to the random indices as centroids
    centroids = X[random_indices]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float) 

def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    distances = np.zeros((k, X.shape[0]))

    for i, centroid in enumerate(centroids):
        distance = np.sum(np.abs(X - centroid) ** p, axis=1)
        distances[i] = distance **(1/p)

    return distances
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distances

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    num_pixels = X.shape[0]
    prev_classes = np.zeros(num_pixels, dtype=int)

    for iteration in range(max_iter):
        # Step 1: Assign each RGB point to its closest centroid
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)

        # Step 2: Calculate new centroids
        new_centroids = np.array([X[classes == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(classes == prev_classes):
            break

        centroids = new_centroids
        prev_classes = classes
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None
    centroids = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    num_pixels = X.shape[0]

    # Step 1: Choose the first centroid uniformly at random among the data points.
    initial_centroid_idx = np.random.choice(num_pixels, size=1)
    centroids = X[initial_centroid_idx]

    for _ in range(k - 1):
        # Step 2: Compute the distance between each data point and the nearest centroid.
        distances = lp_distance(X, centroids, p)
        min_distances = np.min(distances, axis=0)

        # Step 3: Choose a new data point as a new centroid using weighted probability.
        squared_probabilities_sum = np.sum(min_distances ** 2)
        probabilities = (min_distances ** 2) / squared_probabilities_sum
        new_centroid_idx = np.random.choice(num_pixels, size=1, p=probabilities)
        centroids = np.vstack((centroids, X[new_centroid_idx]))

    # Run standard k-means using the initial centroids obtained from k-means++.
    prev_classes = np.zeros(num_pixels, dtype=int)

    for iteration in range(max_iter):
        # Step 1: Assign each RGB point to its closest centroid
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)

        # Step 2: Calculate new centroids
        new_centroids = np.array([X[classes == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(classes == prev_classes):
            break

        centroids = new_centroids
        prev_classes = classes

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes






