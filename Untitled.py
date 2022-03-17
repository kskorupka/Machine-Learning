#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
size = 300
X = np.random.rand(size)*5-2.5
w4,w3,w2,w1,w0 = 1,2,1,-4,2
y=w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.to_csv('dane_do_regresji.csv', index=None)
df.plot.scatter(x='x', y='y')


# In[39]:


from sklearn.model_selection import train_test_split


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(df[['x']], df[['y']], test_size=0.2, random_state=42)


# In[45]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)


# In[ ]:




