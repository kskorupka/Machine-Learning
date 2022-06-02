#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()


# In[17]:


def build_model(n_hidden, n_neurons, optimizer, learning_rate, momentum=0):
    model = tf.keras.models.Sequential()
    for i in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons,activation="relu"))
    if optimizer in ["SGD","sgd"]:
        opt = optimizer=tf.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer=="nesterov":
        opt = optimizer=tf.optimizers.SGD(learning_rate=learning_rate, nesterov=True, momentum=momentum)
    elif optimizer=="momentum":
        opt = optimizer=tf.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    else:
        opt = opt = optimizer=tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=opt, metrics=["mae"])
    return model


# In[26]:


# 100 epochs
# jesli sie nie nauczy, to (NaN, NaN, Nan)


# In[29]:


n_hidden = 1


# In[30]:


learning_rate = 10**(-5)


# In[31]:


n_neurons = 25


# In[32]:


opt = optimizer=tf.optimizers.SGD(learning_rate=learning_rate)


# In[33]:


base_model = build_model(n_hidden,n_neurons,opt,learning_rate)


# In[34]:


lr = [10**(-6),10**(-5),10**(-4)]


# In[35]:


hl = [0,1,2,3]


# In[36]:


nn = [5,25,125]


# In[37]:


mom = [0.1, 0.5, 0.9]


# In[25]:


# base_model.fit(X_train,y_train,epochs=100)


# In[39]:


# def function to get the directiory


# In[38]:


for value in lr:
    learning_rate = value
    model = build_model(n_hidden,n_neurons,opt,learning_rate)
    es = tf.keras.callbacks.EarlyStopping(patience=10,min_delta=1.00,verbose=1)
    # tb = tf.keras.callbacks.TensorBoard(# use that function to  get directory)


# In[ ]:




