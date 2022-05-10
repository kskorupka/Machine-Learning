from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN

mnist = fetch_openml('mnist_784', version=1, as_frame=False) 
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]

kmeans_8 = KMeans(n_clusters=8,random_state=42)
kmeans_9 = KMeans(n_clusters=9,random_state=42)
kmeans_10 = KMeans(n_clusters=10,random_state=42)
kmeans_11 = KMeans(n_clusters=11,random_state=42)
kmeans_12 = KMeans(n_clusters=12,random_state=42)

y_pred_8 = kmeans_8.fit_predict(X)
y_pred_9 = kmeans_9.fit_predict(X)
y_pred_10 = kmeans_10.fit_predict(X)
y_pred_11 = kmeans_11.fit_predict(X)
y_pred_12 = kmeans_12.fit_predict(X)

sil_score_8 = silhouette_score(X,kmeans_8.labels_)
sil_score_9 = silhouette_score(X,kmeans_9.labels_)
sil_score_10 = silhouette_score(X,kmeans_10.labels_)
sil_score_11 = silhouette_score(X,kmeans_11.labels_)
sil_score_12 = silhouette_score(X,kmeans_12.labels_)

kmeans_sil = [sil_score_8, sil_score_9, sil_score_10, sil_score_11, sil_score_12]

with open('kmeans_sil.pkl','wb') as f:
    pickle.dump(kmeans_sil,f)

i = 8
for x in kmeans_sil:
    print(i,": ",x)
    i+=1

with open('kmeans_sil.pkl','rb') as f:
    kmeans_sil_test = pickle.load(f)

print(kmeans_sil_test)

#Najlepszy wynik dla n=8

conf_matrix = confusion_matrix(y_pred_10,y)

print(conf_matrix)

kmeans_argmax = []

for row in conf_matrix:
    max = np.argmax(row)
    print(max)
    print(row)
    kmeans_argmax.append(max)

print(kmeans_argmax)

kmeans_argmax_set_list = [x for x in set(kmeans_argmax)]

print(kmeans_argmax_set_list)

with open('kmeans_argmax.pkl','wb') as f:
    pickle.dump(kmeans_argmax_set_list,f)

with open('kmeans_argmax.pkl','rb') as f:
    kmeans_argmax_test = pickle.load(f)

print(kmeans_argmax_test)

eps = []

for i,x in enumerate(X[:300]):
    for y in X[i+1:300]:
        eps.append(np.linalg.norm(x-y))

eps = sorted(eps)

print(eps[:10])

dist = eps[:10]

with open('dist.pkl','wb') as f:
    pickle.dump(dist,f)

with open('dist.pkl','rb') as f:
    dist_test = pickle.load(f)

print(dist)

print(dist_test)

s = (dist[0]+dist[1]+dist[2])/3

max_s = 1.1*s

dbscan_len = []

while s <= max_s:
    dbscan = DBSCAN(eps = s)
    predict = dbscan.fit_predict(X)
    dbscan_len.append(len(set(predict)))
    s = 1.04*s

print(dbscan_len)

with open('dbscan_len.pkl','wb') as f:
    pickle.dump(dbscan_len,f)

with open('dbscan_len.pkl','rb') as f:
    dbscan_len_test = pickle.load(f)

print(dbscan_len_test)