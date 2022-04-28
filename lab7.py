#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml 
import numpy as np


# In[2]:


mnist = fetch_openml('mnist_784', version=1, as_frame=False) 
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]


# In[3]:


from sklearn.cluster import KMeans


# In[4]:


kmeans_8 = KMeans(n_clusters=8,random_state=42)
kmeans_9 = KMeans(n_clusters=9,random_state=42)
kmeans_10 = KMeans(n_clusters=10,random_state=42)
kmeans_11 = KMeans(n_clusters=11,random_state=42)
kmeans_12 = KMeans(n_clusters=12,random_state=42)


# In[5]:


y_pred_8 = kmeans_8.fit_predict(X)
y_pred_9 = kmeans_9.fit_predict(X)
y_pred_10 = kmeans_10.fit_predict(X)
y_pred_11 = kmeans_11.fit_predict(X)
y_pred_12 = kmeans_12.fit_predict(X)


# In[6]:


from sklearn.metrics import silhouette_score


# In[7]:


sil_score_8 = silhouette_score(X,kmeans_8.labels_)
sil_score_9 = silhouette_score(X,kmeans_9.labels_)
sil_score_10 = silhouette_score(X,kmeans_10.labels_)
sil_score_11 = silhouette_score(X,kmeans_11.labels_)
sil_score_12 = silhouette_score(X,kmeans_12.labels_)


# In[8]:


kmeans_sil = [sil_score_8, sil_score_9, sil_score_10, sil_score_11, sil_score_12]


# In[9]:


import pickle


# In[10]:


with open('kmeans_sil.pkl','wb') as f:
    pickle.dump(kmeans_sil,f)


# In[11]:


i = 8
for x in kmeans_sil:
    print(i,": ",x)
    i+=1


# In[12]:


with open('kmeans_sil.pkl','rb') as f:
    kmeans_sil_test = pickle.load(f)


# In[13]:


print(kmeans_sil_test)


# In[14]:


#Najlepszy wynik dla n=8


# In[15]:


from sklearn.metrics import confusion_matrix


# In[16]:


conf_matrix = confusion_matrix(y_pred_10,y)


# In[17]:


conf_matrix


# In[18]:


kmeans_argmax = []


# In[19]:


for row in conf_matrix:
    max = np.argmax(row)
    print(max)
    print(row)
    kmeans_argmax.append(max)


# In[20]:


kmeans_argmax


# In[24]:


kmeans_argmax_set_list = [x for x in set(kmeans_argmax)]


# In[25]:


kmeans_argmax_set_list


# In[27]:


with open('kmeans_argmax.pkl','wb') as f:
    pickle.dump(kmeans_argmax_set_list,f)


# In[28]:


with open('kmeans_argmax.pkl','rb') as f:
    kmeans_argmax_test = pickle.load(f)


# In[29]:


print(kmeans_argmax_test)


# In[ ]:




