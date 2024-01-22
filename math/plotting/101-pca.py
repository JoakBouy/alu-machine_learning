#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

iris_dict = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

lib = np.load("/content/data.npy")
data = lib
lib_labels = np.load("/content/labels.npy")
labels = lib_labels

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(3):
    ax.scatter(pca_data[labels == i, 0], pca_data[labels == i, 1], pca_data[labels == i, 2], cmap='plasma', label=iris_dict[i])

ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
ax.set_title('PCA of Iris Dataset')
ax.legend()
plt.show()