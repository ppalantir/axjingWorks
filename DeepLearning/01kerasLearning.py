'''
@Description: In User Settings Edit
@Author: XJing An
@Date: 2019-10-06 16:15:41
@LastEditTime: 2019-10-06 16:42:01
@LastEditors: Please set LastEditors
'''
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np 

model = tf.keras.Sequential()
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))



# add a debsely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
#add another
model.add(layers.Dense(64, activation='relu'))
#Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation="softmax"))

model.compile(optimizer=tf.keras.optimizer.Adam(0.01),
                loss=tf.keras.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalCrossentropy()])
                

model.fit(data, labels, epochs=10, batch_size=32)

