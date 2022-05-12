#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris 
iris = load_iris(as_frame=True)


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


pd.concat([iris.data, iris.target], axis=1).plot.scatter(
    x='petal length (cm)',
    y='petal width (cm)',
    c='target',
    colormap='viridis'
)


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


#X_train, X_test, y_train, y_test = train_test_split(iris['data'][:,(2,3)],iris.target, test_size=0.2)


# In[6]:


X = iris.data.to_numpy()[:, (2,3)]


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X,iris.target,test_size=0.2)


# In[8]:


y_0_train = (y_train == 0).astype(int)
y_1_train = (y_train == 1).astype(int)
y_2_train = (y_train == 2).astype(int)


# In[9]:


y_0_test = (y_test == 0).astype(int)
y_1_test = (y_test == 1).astype(int)
y_2_test = (y_test == 2).astype(int)


# In[10]:


from sklearn.linear_model import Perceptron


# In[11]:


per_clf_0 = Perceptron()
per_clf_1 = Perceptron()
per_clf_2 = Perceptron()


# In[12]:


per_clf_0.fit(X_train,y_0_train)
per_clf_1.fit(X_train,y_1_train)
per_clf_2.fit(X_train,y_2_train)


# In[13]:


from sklearn.metrics import accuracy_score


# In[14]:


ac_0_train = accuracy_score(y_0_train, per_clf_0.predict(X_train))
ac_0_test = accuracy_score(y_0_test, per_clf_0.predict(X_test))


# In[15]:


ac_1_train = accuracy_score(y_1_train, per_clf_1.predict(X_train))
ac_1_test = accuracy_score(y_1_test, per_clf_1.predict(X_test))


# In[16]:


ac_2_train = accuracy_score(y_2_train, per_clf_2.predict(X_train))
ac_2_test = accuracy_score(y_2_test, per_clf_2.predict(X_test))


# In[17]:


accuracy_list = [(ac_0_train,ac_0_test),(ac_1_train,ac_1_test),(ac_2_train,ac_2_test)]


# In[18]:


w_0_0 = per_clf_0.intercept_
w_0_1,w_0_2 = per_clf_0.coef_[0][0],per_clf_0.coef_[0][1]


# In[19]:


w_1_0 = per_clf_1.intercept_
w_1_1,w_1_2 = per_clf_1.coef_[0][0],per_clf_1.coef_[0][1]


# In[20]:


w_2_0 = per_clf_2.intercept_
w_2_1,w_2_2 = per_clf_2.coef_[0][0],per_clf_2.coef_[0][1]


# In[21]:


weight_list = [(w_0_0, w_0_1, w_0_2),(w_1_0,w_1_1,w_1_2),(w_2_0,w_2_1,w_2_2)]


# In[22]:


import pickle


# In[23]:


with open('per_acc.pkl','wb') as f:
    pickle.dump(accuracy_list,f)


# In[24]:


with open('per_wght.pkl','wb') as f:
    pickle.dump(weight_list,f)


# In[25]:


X_2 = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y_2 = np.array([0,
              1,
              1,
              0])


# In[26]:


per_clf = Perceptron()


# In[27]:


per_clf.fit(X_2,y_2)


# In[28]:


ac_2 = accuracy_score(y_2, per_clf.predict(X_2))


# In[29]:


print(ac_2)

