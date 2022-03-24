#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
data = datasets.load_breast_cancer()
print(data['DESCR'])


# In[2]:


import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# In[3]:


X = data.data[:,(3,5)]
y = data.target


# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


svm_clf_without_scaler = Pipeline([("linear_svc", LinearSVC(C=1,loss="hinge",random_state=42)),])


# In[6]:


svm_clf_with_scaler = Pipeline([("scaler", StandardScaler()),("linear_svc", LinearSVC(C=1,loss="hinge",
                                random_state=42)),])


# In[7]:


svm_clf_without_scaler.fit(X_train,y_train)
svm_clf_with_scaler.fit(X_train,y_train)


# In[8]:


acc_svm_unscaled_train = svm_clf_without_scaler.score(X_train, y_train)
acc_svm_unscaled_test = svm_clf_without_scaler.score(X_test, y_test)
acc_svm_scaled_train = svm_clf_with_scaler.score(X_train, y_train)
acc_svm_scaled_test = svm_clf_with_scaler.score(X_test, y_test)


# In[9]:


print(acc_svm_unscaled_train)


# In[10]:


print(acc_svm_unscaled_test)


# In[11]:


print(acc_svm_scaled_train)


# In[12]:


print(acc_svm_scaled_test)


# In[13]:


bc_acc = [acc_svm_unscaled_train, acc_svm_unscaled_test, acc_svm_scaled_train, acc_svm_scaled_test]


# In[14]:


print(bc_acc)


# In[15]:


import pickle
with open('bc_acc.pkl', 'wb') as f:
    pickle.dump(bc_acc, f)


# In[16]:


data_iris = datasets.load_iris()
print(data_iris['DESCR'])


# In[17]:


print(data_iris.feature_names)


# In[33]:


X_iris = data_iris.data[: ,(2,3)]
y_iris = (data_iris["target"] == 2).astype(np.int8)


# In[34]:


X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2)


# In[35]:


iris_svm_clf_without_scaler = Pipeline([("linear_svc", LinearSVC(C=1,loss="hinge",random_state=42)),])


# In[36]:


iris_svm_clf_with_scaler = Pipeline([("scaler", StandardScaler()),("linear_svc", LinearSVC(C=1,loss="hinge",
                                random_state=42)),])


# In[37]:


iris_svm_clf_without_scaler.fit(X_train_iris,y_train_iris)
iris_svm_clf_with_scaler.fit(X_train_iris,y_train_iris)


# In[38]:


iris_acc_svm_unscaled_train = iris_svm_clf_without_scaler.score(X_train_iris, y_train_iris)
iris_acc_svm_unscaled_test = iris_svm_clf_without_scaler.score(X_test_iris, y_test_iris)
iris_acc_svm_scaled_train = iris_svm_clf_with_scaler.score(X_train_iris, y_train_iris)
iris_acc_svm_scaled_test = iris_svm_clf_with_scaler.score(X_test_iris, y_test_iris)


# In[39]:


iris_acc = [iris_acc_svm_unscaled_train,iris_acc_svm_unscaled_test,iris_acc_svm_scaled_train,iris_acc_svm_scaled_test]


# In[40]:


print(iris_acc)


# In[ ]:




