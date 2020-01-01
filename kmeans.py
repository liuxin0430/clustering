"""
clusters the list of points into `k` clusters using k-means clustering algorithm.
"""

import numpy as np

# sum of squared errors for the given list of data points to the centroid
def sse(points):
    centroid = np.mean(points, 0)
    errors = np.linalg.norm(points-centroid, ord=2, axis=1)
    return np.sum(errors)

def kmeans(points, K = 4, max_iter = 100):
    if len(points) < K:
        print("[Error]: Number of data points can't be less than k")
        return None
    epochs = 10
    label = np.zeros(len(points))
    best_label = None
    for ep in range(epochs):
        # randomly initialize k centroids
        cen_idx = np.random.randint(low=0,high=len(points),size=K)
        centroids = points[cen_idx,:]

        best_sse = np.inf
        last_sse = np.inf
        it = 0
        while it < max_iter:
            for i in range(len(points)):
                point = points[i]
                index = np.argmin(np.linalg.norm(centroids-point, 2, 1))
                label[i] = index+1
            cur_sse = 0
            for i in range(K):
                clustered_points = points[np.where(label == (i+1))]   
                centroids[i] = np.mean(clustered_points, 0)
                cur_sse += sse(clustered_points) #SSE calculation

            gain = last_sse - cur_sse

            # Check for improvement
            if cur_sse < best_sse:
                best_sse = cur_sse
                best_label = label

            # Epoch termination condition
            if np.isclose(gain, 0, atol=0.00001):
                break
            last_sse = cur_sse
            it += 1
    return best_label