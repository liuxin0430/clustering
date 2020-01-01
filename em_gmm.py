'''
EM_GMM
Expectation Maximization Algorithm

'''

'''
n: the number of points
d: the number of features for each point
K: the number of clusters
'''
'''
parameters = [pi, mu, sigma]
pi: K*1
mu: K*d
sigma: K*(d*d) diagonal
'''

import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
import copy
import math

# the probability of a point coming from a Guassian with given parameters
## note that the covariance must be diagonal for this to work
def prob(val, mu, sig, pi):
  p = pi
  # val is data point value
  for i in range(len(val)):
    
    temp = norm.pdf(val[i], mu[i], sig[i][i])
    #if math.isnan(temp):
     # print(temp) 
    p *= temp
  #print(p)
  return p


# expectaion step
def expectation(points,  K, parameters):
  pi = parameters[0]
  mu = parameters[1]
  sig = parameters[2]
  #print(sig)
  new_label = np.zeros(len(points))

  for i in range(len(points)):
    point = points[i]
    p_cluster = []
    for j in range(K):
      p_j = prob(point, mu[j], sig[j], pi[j])
      p_cluster.append(p_j)
    p_cluster = np.asarray(p_cluster)
    #print(p_cluster)
    new_label[i] = np.argmax(p_cluster) + 1
    #print(new_label[i])

  # to ensure we hava K labeled data points
  new_label_num = len(np.unique(new_label))
  while new_label_num != K:
    #print("bad!")
    #print("label_num = {}".format(new_label_num))
    new_label = np.random.randint(low=1,high=K+1,size=(len(points)))
    new_label_num = len(np.unique(new_label))
  
  return new_label

# maximization step
# update estimates of pi, mu and sigma
def maximization(points, label, K, parameters):
  new_params = copy.deepcopy(parameters)

  for i in range(K):
    clustered_points = points[np.where(label == (i+1))]
    if len(clustered_points)==0:
      print("i: %s" % i)
      print(clustered_points.shape)
    updated_pi =  len(clustered_points) / float(len(points))
    new_params[0][i] = updated_pi

    updated_mu = np.mean(clustered_points, axis=0)
    new_params[1][i] = updated_mu

    updated_sig_ele = np.std(clustered_points, axis=0)
    
    if math.isnan(updated_sig_ele[1]):
      pass
      #print("liuxin")
      #print(clustered_points)
      
    updated_sig = np.diag(updated_sig_ele)
    new_params[2][i] = updated_sig
  return new_params

# get the distance between points
# used for determining if params have converged
def distance(old_params, new_params):
  dist = 0
  old_mu = old_params[1]
  new_mu = new_params[1]

  diff = new_mu - old_mu

  dist = (diff**2).sum()
  return dist ** 0.5

# loop until parameters converge
'''
parameters = [pi, mu, sigma]
pi: K dimension
mu: K*d
sigma: K*(d*d) diagonal
'''

def em_gmm(points, K=4, d=2):

  # initial guesses - intentionally bad
  guess_pi = np.ones(K)/K

  #guess_mu = np.array([[-5,5], [5,5], [-5,-5], [5,-5]])
  #guess_mu = np.array([[-6,0], [-2,0], [2,0], [5,0]])
  guess_mu = np.random.randint(low = points.min(), high= points.max(), size=(K,d))

  guess_sig = np.ones(d)*10
  #guess_sig = np.random.randint(low = 1, high = 9, size=d)
  guess_sig = np.diag(guess_sig)
  guess_sig = guess_sig.ravel()
  guess_sig = np.tile(guess_sig, K)
  guess_sig = guess_sig.reshape(K,d,d)

  guess_params = [guess_pi, guess_mu, guess_sig]
  params = guess_params

  label = np.zeros(len(points))
  shift = sys.maxsize
  epsilon = 0.01
  iters = 0

  while shift > epsilon:
    
    iters += 1
    # E-step
    updated_label = expectation(points, K, params)  
    # M-step
    updated_parameters = maximization(points, updated_label, K, params)

    # see how much updated mu have changed
    shift = distance(params, updated_parameters)

    # logging
    #print("iteration {}, shift {}".format(iters, shift))

    # update labels and params for the next iteration
    label = updated_label
    params = updated_parameters
  #print("iteration: {}".format(iters))
  return label
