'''
mean shift algorithm

'''
import math
import numpy as np
import sys
import matplotlib.pyplot as plt

# euclidean distance between two n-dimension points
def euclidean_dist(pointA, pointB):
    if(len(pointA) != len(pointB)):
        raise Exception("point dimensionality is not matched!")
    total = float(0)
    for dimension in range(0, len(pointA)):
        total += (pointA[dimension] - pointB[dimension])**2
    dist = math.sqrt(total)
    return dist


def gaussian_kernel(distance, bandwidth):
    euclidean_distance = np.sqrt(((distance)**2).sum(axis=1))
    gaussian_val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((euclidean_distance / bandwidth))**2)
    return gaussian_val


def multivariate_gaussian_kernel(distances, bandwidths):
    # Number of dimensions of the multivariate gaussian
    dim = len(bandwidths)

    # Covariance matrix
    cov = np.multiply(np.power(bandwidths, 2), np.eye(dim))

    # Compute Multivariate gaussian (vectorized implementation)
    exponent = -0.5 * np.sum(np.multiply(np.dot(distances, np.linalg.inv(cov)), distances), axis=1)
    multi_gaussian_val = (1 / np.power((2 * math.pi), (dim/2)) * np.power(np.linalg.det(cov), 0.5)) * np.exp(exponent)

    return multi_gaussian_val

GROUP_DISTANCE_TOLERANCE = .1
def distance_to_group(point, group):
    min_distance = sys.float_info.max
    for pt in group:
        dist = euclidean_dist(point, pt)
        if dist < min_distance:
            min_distance = dist
    return min_distance

def determine_nearest_group(point, groups):
    nearest_group_index = None
    index = 0
    for group in groups:
        dist2group = distance_to_group(point, group)
        if dist2group < GROUP_DISTANCE_TOLERANCE:
            nearest_group_index = index
        index += 1
    return nearest_group_index



def group_points(points):
    group_assignment = []
    groups = []
    group_index = 0
    for point in points:
        nearest_group_index = determine_nearest_group(point, groups)
        if nearest_group_index is None:
            # create new group
            groups.append([point])
            group_assignment.append(group_index)
            group_index += 1
        else:
            group_assignment.append(nearest_group_index)
            groups[nearest_group_index].append(point)
    return np.array(group_assignment)



def shift_point(point, points, kernel_bandwidth):
    
    if isinstance(kernel_bandwidth, int):
        kernel = gaussian_kernel #default gaussian kernel
    else:
        kernel = multivariate_gaussian_kernel
      
    points = np.array(points)
    # numerator
    point_weights = kernel(point-points, kernel_bandwidth)
    tiled_weights = np.tile(point_weights, [len(point), 1])
    # denominator
    denominator = sum(point_weights)
    shifted_point = np.multiply(tiled_weights.transpose(), points).sum(axis=0) / denominator
    return shifted_point

MIN_DISTANCE = 0.000001
def mean_shift(points, kernel_bandwidth):
    shift_points = np.array(points)
    max_min_dist = 1
    #iteration_number = 0

    still_shifting = [True] * points.shape[0]
    while max_min_dist > MIN_DISTANCE:
        max_min_dist = 0
        #iteration_number = iteration_number + 1
        for i in range(0, len(shift_points)):
            if not still_shifting[i]:
                continue
            p_new = shift_points[i]
            p_new_start = p_new
            p_new = shift_point(p_new, points, kernel_bandwidth)
            dist = euclidean_dist(p_new, p_new_start)

            if dist > max_min_dist:
                max_min_dist = dist
            if dist < MIN_DISTANCE:
                still_shifting[i] = False
            shift_points[i] = p_new

    label = group_points(shift_points.tolist()) + 1
    return label

