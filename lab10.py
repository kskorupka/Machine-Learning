#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data() 
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)


# In[18]:


import numpy as np


# In[6]:


print(type(X_test))


# In[7]:


X_train = X_train.astype('float')


# In[8]:


X_test = X_test.astype('float')


# In[15]:


for x in X_train:
    x/=255


# In[20]:


for x in X_test:
    x/=255


# In[23]:


import matplotlib.pyplot as plt 
plt.imshow(X_train[142], cmap="binary") 
plt.axis('off')
plt.show()


# In[25]:


class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka",
               "sandał", "koszula", "but", "torba", "kozak"]
class_names[y_train[142]]


# In[26]:


from tensorflow import keras


# In[48]:


model = keras.models.Sequential()


# In[49]:


model.add(keras.layers.Flatten(input_shape =[28,28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10,activation="softmax"))


# In[50]:


model.summary()
tf.keras.utils.plot_model(model, "fashion_mnist.png", show_shapes=True)


# In[58]:


model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])


# In[59]:


import os
root_logdir = os.path.join(os.curdir, "image_logs")
def get_run_logdir(): 
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S") 
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[63]:


history = model.fit(X_train,y_train,epochs=20, validation_split=0.1)


# In[ ]:




