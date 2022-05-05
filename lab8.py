#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer()


# In[73]:


import numpy as np


# In[2]:


from sklearn.datasets import load_iris 
data_iris = load_iris()


# In[3]:


from sklearn.decomposition import PCA


# In[32]:


pca_db = PCA(n_components=0.9)


# In[33]:


pca_di = PCA(n_components=0.9)


# In[34]:


X_reduced_db = pca_db.fit_transform(data_breast_cancer.data)


# In[35]:


X_reduced_di = pca_di.fit_transform(data_iris.data)


# In[36]:


print(X_reduced_db.shape)


# In[37]:


print(data_breast_cancer.data.shape)


# In[38]:


print(X_reduced_di.shape)


# In[39]:


print(data_iris.data.shape)


# In[40]:


from sklearn.preprocessing import StandardScaler


# In[41]:


sca = StandardScaler()


# In[42]:


sca.fit(data_breast_cancer.data)


# In[43]:


X_scaled_db = sca.transform(data_breast_cancer.data)


# In[44]:


sca.fit(data_iris.data)


# In[47]:


pca_db_scaled = PCA(n_components=0.9)


# In[48]:


pca_di_scaled = PCA(n_components=0.9)


# In[49]:


X_scaled_di = sca.transform(data_iris.data)


# In[50]:


X_reduced_db_scaled = pca_db_scaled.fit_transform(X_scaled_db)


# In[51]:


X_reduced_di_scared = pca_di_scaled.fit_transform(X_scaled_di)


# In[52]:


print(X_reduced_db_scaled.shape)


# In[58]:


print(X_reduced_di_scared.shape)


# In[59]:


import pickle


# In[60]:


with open('pca_bc.pkl','wb') as f:
    pickle.dump(pca_db_scaled.explained_variance_ratio_,f)


# In[61]:


with open('pca_ir.pkl','wb') as f:
    pickle.dump(pca_di_scaled.explained_variance_ratio_,f)


# In[64]:


with open('pca_bc.pkl','rb') as f:
    bc_test_list = pickle.load(f)


# In[65]:


with open('pca_ir.pkl','rb') as f:
    ir_test_list = pickle.load(f)


# In[66]:


print(bc_test_list)


# In[67]:


print(ir_test_list)


# In[78]:


print(pca_di_scaled.components_)


# In[81]:


di_idx = []
db_idx = []


# In[83]:


for row in pca_di_scaled.components_:
    di_idx.append(np.argmax(row))
    print(np.max(row),np.argmax(row))


# In[84]:


for row in pca_db_scaled.components_:
    db_idx.append(np.argmax(row))
    print(np.max(row),np.argmax(row))


# In[85]:


print(di_idx)


# In[86]:


print(db_idx)


# In[87]:


with open('idx_bc.pkl','wb') as f:
    pickle.dump(db_idx,f)


# In[88]:


with open('idx_ir.pkl','wb') as f:
    pickle.dump(di_idx,f)


# In[89]:


with open('idx_bc.pkl','rb') as f:
    idx_bc_test = pickle.load(f)


# In[90]:


with open('idx_ir.pkl','rb') as f:
    idx_ir_test = pickle.load(f)


# In[91]:


print(idx_bc_test)
print(idx_ir_test)


# In[ ]:




