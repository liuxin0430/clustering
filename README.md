Implementation of Clustering Algorithms, including K-means algorithm, EM algorithm for Gaussian mixture models(EM-GMM) and Mean-shift algorithm.

Python version: 3.7.3


notes:
1) The three clustering algorithms are wrapped as functions, you can use them easily.
2) Run run_cluster_alg.py can get each clustering result of input data.
3) In run_cluster_alg.py, the "data path" should be changed according to your directory manually (like line 28-30 of run_cluster_alg.py)

***
### input and output for each clustering algorithm function

#### kmeans
![kmeans](https://github.com/liuxin0430/image_folder/blob/master/kmeans_alg.JPG)

input: 
* points : each row is a data point
* K = 4: the number of clusters, default 4
* max_iter = 100: max iteration number, default 100

output:
* label: the label of each data point


#### em_gmm
![em_gmm](https://github.com/liuxin0430/image_folder/blob/master/em_gmm_alg.JPG)

input:
* points: : each row is a data point
* K: the number of clusters, default 4
* d: the dimension of data point, default 2

output:
* label: the label of each data point

#### mean_shift
![mean_shift](https://github.com/liuxin0430/image_folder/blob/master/mean_shift_alg.JPG)

input:
* points: each row is a data point
* kernel_bandwidth: kernel bandwidth

output:
* label: the label of each data point