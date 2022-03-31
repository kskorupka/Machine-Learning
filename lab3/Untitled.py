#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[4]:


import numpy as np
import pandas as pd
size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4 
df = pd.DataFrame({'x': X, 'y': y})


# In[6]:


df.plot.scatter(x='x',y='y')


# In[7]:


from sklearn.model_selection import train_test_split


# In[10]:


X_bc_train, X_bc_test, y_bc_train, y_bc_test = train_test_split(data_breast_cancer.data, data_breast_cancer.target, test_size=0.2, random_state=42)


# In[11]:


from sklearn.tree import DecisionTreeClassifier


# In[116]:


tree_clf_bc = DecisionTreeClassifier(max_depth=1, random_state=42)


# In[117]:


tree_clf_bc.fit(X_bc_train, y_bc_train)


# In[118]:


from sklearn.tree import export_graphviz


# In[119]:


f = "bc_tree.dot"


# In[120]:


export_graphviz(tree_clf_bc, out_file=f, feature_names=data_breast_cancer.feature_names, 
                class_names=data_breast_cancer.target_names, rounded=True, filled=True)
print(f)


# In[121]:


import graphviz
print(graphviz.render('dot', 'png', f))


# In[122]:


graph = graphviz.Source.from_file(f)


# In[123]:


graph


# In[124]:


f1_list = []


# In[125]:


from sklearn.metrics import f1_score


# In[126]:


y_bc_pred_train = tree_clf_bc.predict(X_bc_train)


# In[127]:


y_bc_pred_test = tree_clf_bc.predict(X_bc_test)


# In[128]:


f1_list.append((f1_score(y_bc_train, y_bc_pred_train), f1_score(y_bc_test, y_bc_pred_test)))


# In[129]:


print(f1_list)


# In[130]:


best_depth = 1


# In[131]:


for depth in range(2,20):
    tree_clf_bc = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_clf_bc.fit(X_bc_train, y_bc_train)
    y_bc_pred_train = tree_clf_bc.predict(X_bc_train)
    y_bc_pred_test = tree_clf_bc.predict(X_bc_test)
    f1_bc_score_train = f1_score(y_bc_train, y_bc_pred_train)
    f1_bc_score_test = f1_score(y_bc_test, y_bc_pred_test)
    if (f1_bc_score_train, f1_bc_score_test) > max(f1_list):
        best_depth = depth
    f1_list.append((f1_bc_score_train, f1_bc_score_test))


# In[132]:


print(f1_list)


# In[133]:


print(best_depth)


# In[134]:


for x in f1_list:
    print(x)


# In[ ]:




