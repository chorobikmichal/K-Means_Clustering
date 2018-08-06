# K-Means Clustering

#importing librarys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the mall dataset with pandas
dataSet = pd.read_csv('mall.csv')
x = dataSet.iloc[:,[3,4]].values

#using the elbow method to determine the optimal amount of clusters(it's 5)
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)#calculates the sum of sqares and adds it to wcss
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('number of clusters')
plt.xlabel('wcss')
plt.show()

#applying kmeans to the mall dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

#visulizing the clusters
plt.scatter(x[y_kmeans == 0,0],x[y_kmeans == 0,1],s=100,c='red', label='Cluster 1')
plt.scatter(x[y_kmeans == 1,0],x[y_kmeans == 1,1],s=100,c='blue', label='Cluster 2')
plt.scatter(x[y_kmeans == 2,0],x[y_kmeans == 2,1],s=100,c='green', label='Cluster 3')
plt.scatter(x[y_kmeans == 3,0],x[y_kmeans == 3,1],s=100,c='cyan', label='Cluster 4')
plt.scatter(x[y_kmeans == 4,0],x[y_kmeans == 4,1],s=100,c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300,c='yellow',label='Centroids')
plt.title('Clusters of slients')
plt.xlabel('Anual income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()