#!/usr/bin/env python
# coding: utf-8

# In[20]:


import tensorflow_datasets as tfds
[test_set_raw, valid_set_raw, train_set_raw], info = tfds.load(
    "tf_flowers",
split=["train[:10%]", "train[10%:25%]", "train[25%:]"], as_supervised=True,
with_info=True)


# In[21]:


# maxpool2d


# In[22]:


# 32 64 128
# 64 128 256
# 10 neuronow i softmax na wyjsciu


# In[23]:


info


# In[24]:


class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples


# In[25]:


import matplotlib.pyplot as plt


# In[26]:


plt.figure(figsize=(12, 8))
index = 0
sample_images = train_set_raw.take(9) 
for image, label in sample_images:
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title("Class: {}".format(class_names[label])) 
    plt.axis("off")
plt.show(block=False)


# In[27]:


import tensorflow as tf


# In[28]:


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224]) 
    return resized_image, label


# In[29]:


batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# In[30]:


plt.figure(figsize=(8, 8)) 
sample_batch = train_set.take(1)
for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1) 
        plt.imshow(X_batch[index]/255.0)
        plt.title("Class: {}".format(class_names[y_batch[index]])) 
        plt.axis("off")
plt.show()


# In[146]:


model = tf.keras.Sequential()


# In[147]:


model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=7,input_shape=(224,224,3)))


# In[148]:


model.add(tf.keras.layers.MaxPool2D(pool_size=2))


# In[149]:


model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=5))
model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=5))


# In[150]:


model.add(tf.keras.layers.MaxPool2D(pool_size=2))


# In[151]:


model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3))
model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3))


# In[152]:


model.add(tf.keras.layers.MaxPool2D(pool_size=2))


# In[153]:


model.add(tf.keras.layers.Flatten())


# In[154]:


model.add(tf.keras.layers.Dense(units=64,activation='relu'))


# In[155]:


model.add(tf.keras.layers.Dropout(0.5))


# In[156]:


model.add(tf.keras.layers.Dense(units=32,activation='relu'))


# In[157]:


model.add(tf.keras.layers.Dropout(0.5))


# In[158]:


model.add(tf.keras.layers.Dense(units=10,activation='softmax'))


# In[159]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[160]:


history = model.fit(train_set,epochs=10,validation_data=valid_set)


# In[ ]:




