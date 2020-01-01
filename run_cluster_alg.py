'''
Run the three clustering algorithms on the three synthetic datasets
'''
import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans
from mean_shift import mean_shift
from em_gmm import em_gmm

#get data points
def load_data_X(file_name):
    data_X = np.loadtxt(file_name)
    data_X = data_X.T
    return data_X

# scatter plot of the predict label of clustering alg
def visualize_clustering(data_X, label, data_name, alg_name):
    fig = plt.figure()
    plt.scatter(data_X[:,0], data_X[:,1], 24, c=label)
    plt.title(alg_name + " " +data_name)
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.grid(True, linestyle = ':', color = "gray", linewidth = "0.5")
    fig.savefig(alg_name +"_" + data_name + ".png")

#get data points and label
dataA_X_file = "cluster_data\\cluster_data_dataA_X.txt"
dataB_X_file = "cluster_data\\cluster_data_dataB_X.txt"
dataC_X_file = "cluster_data\\cluster_data_dataC_X.txt"
dataA_X = load_data_X(dataA_X_file)
dataB_X = load_data_X(dataB_X_file)
dataC_X = load_data_X(dataC_X_file)

K = 4
d = 2
'''
label_km = kmeans(dataA_X)
label_em = em_gmm(dataA_X, K, d)
label_ms = mean_shift(dataA_X,3)
'''

data_names = ["dataA", "dataB", "dataC"]
data_Xs = [dataA_X, dataB_X, dataC_X]

# kmeans dataA/ dataB/ dataC
for i in range(3):
    label_km = kmeans(data_Xs[i], K=4)
    visualize_clustering(data_Xs[i], label_km, data_names[i], "kmeans")

# em_gmm dataA/ dataB/ dataC
for i in range(3):
    label_em = em_gmm(data_Xs[i],K=4,d=2)
    visualize_clustering(data_Xs[i], label_em, data_names[i], "em_gmm")

# mean shift dataA/ dataB/ dataC
for i in range(3):
    label_ms = mean_shift(data_Xs[i], 2)
    visualize_clustering(data_Xs[i], label_ms, data_names[i], "mean_shift")


