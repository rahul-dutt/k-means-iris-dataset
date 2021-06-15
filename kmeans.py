#Apply Kmeans to Iris data set and determine true positive and false negative values.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.patches as mpatches

iris_data = pd.read_csv('iris.csv')
print(iris_data.head())

iris_data['Class'] = pd.Categorical(iris_data["Class"])
iris_data["Class"] = iris_data["Class"].cat.codes

X = iris_data.values[:, 0:4]
y = iris_data.values[:, 4]

k_means = KMeans(n_clusters=3)
k_means = k_means.fit(X)

# Centroid values
centroids = k_means.cluster_centers_

y_predicted = k_means.predict(X)

target_names = ['Se', 'Ve', 'Vi']
red_patch = mpatches.Patch(color='yellow', label='setosa')
green_patch = mpatches.Patch(color='green', label='versicolour')
blue_patch = mpatches.Patch(color='blue', label='virginica')

colors = np.array(['yellow', 'green', 'blue'])
plt.scatter(X[:, 0], X[:, 1], c=colors[y_predicted])
plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red', label='Centroids')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend(handles=[red_patch, green_patch, blue_patch])
plt.show()

print(classification_report(iris_data['Class'],k_means.labels_,target_names=target_names))

print(confusion_matrix(y_predicted, y))
print(accuracy_score(y_predicted, y))
