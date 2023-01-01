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
plt.show()