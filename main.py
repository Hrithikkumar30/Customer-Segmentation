import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


data = pd.read_csv("Mall_Customers.csv")
X = data.iloc[ :, [3,4]].values
wcss =[]

for i in range(1,11):
    kmeans = KMeans(n_clusters=i , init="k-means++" , random_state=42)
    kmeans.fit(X)
    
    wcss.append(kmeans.inertia_)
    
    
sns.set()
plt.plot( range(1,11 ) ,wcss)
plt.title("the elbow of plot")
plt.xlabel('number of clusters')
plt.ylabel('wcss')
# plt.show()


kmeans = KMeans(n_clusters=5 , init="k-means++" , random_state=42)
Y = kmeans.fit_predict(X)
# print(Y)
plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='blue', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:0] , kmeans.cluster_centers_[:1] , s=100 ,  c='cyan' , label='centroid')

plt.title('customer clusters')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.show()

# plotting all the clusters and their Centroids



# plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()