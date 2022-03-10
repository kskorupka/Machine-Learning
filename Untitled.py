#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', version=1)


# In[2]:


import pandas as pd


# In[3]:


X,y = mnist["data"], mnist["target"]


# In[4]:


print((np.array(X.loc[42]).reshape(28,28)>0).astype(int))


# In[5]:


print(type(y))


# In[6]:


y_sorted = y.sort_values()


# In[7]:


X_sorted = X.reindex(y_sorted.index)


# In[8]:


print(X_sorted)


# In[9]:


print(y_sorted)


# In[10]:


X_train_sorted, X_test_sorted = X_sorted[:56000], X_sorted[56000:]
y_train_sorted, y_test_sorted = y_sorted[:56000], y_sorted[56000:]


# In[11]:


print(X_train_sorted.shape, y_train_sorted.shape)
print(X_test_sorted.shape, y_test_sorted.shape)


# In[12]:


print(np.unique(y_train_sorted))
print(np.unique(y_test_sorted))


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# In[14]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[15]:


print(np.unique(y_train))
print(np.unique(y_test))


# In[29]:


y_train_0 = (y_train == '0')
y_test_0 = (y_test == '0')


# In[30]:


print(np.unique(y_train_0))
print(np.unique(y_test_0))


# In[31]:


from sklearn.linear_model import SGDClassifier 
sgd_clf = SGDClassifier(random_state=42)


# In[32]:


sgd_clf.fit(X_train, y_train_0)


# In[42]:


train_0_predict = sgd_clf.predict(X_train)
test_0_predict = sgd_clf.predict(X_test)

print(y_train_0 == train_0_predict)


# In[33]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy", n_jobs=-1)


# In[ ]:


from sklearn.metrics import precision_score, recall_score

