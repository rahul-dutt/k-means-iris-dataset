import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



iris_data = pd.read_csv('iris.csv')
print(iris_data.head())

iris_data['Class'] = pd.Categorical(iris_data["Class"])
iris_data["Class"] = iris_data["Class"].cat.codes

X = iris_data.values[:, 0:4]
y = iris_data.values[:, 4]


kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)
# Getting the cluster labels

# Centroid values
centroids = kmeans.cluster_centers_

kmeans = kmeans.predict(X)

target_names = ['Se', 'Ve', 'Vi']

plt.scatter(X[:, 0], X[:, 1], c=kmeans, cmap='rainbow')
plt.show()

# print(classification_report(iris_data['Class'], kmeans.labels_, target_names=target_names))


print(confusion_matrix(kmeans, y))
print(accuracy_score(kmeans, y))
