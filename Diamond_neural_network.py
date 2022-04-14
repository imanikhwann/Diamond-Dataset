# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 12:27:34 2022

@author: owner
"""

import numpy as np 
import pandas as pd 
import tensorflow as tf

data = pd.read_csv(r"C:\Users\owner\Desktop\SHRDC MIDA AIML\Deep Learning\Git Repo\Diamond Dataset\diamonds.csv", index_col=(0))
#%%
x = data.drop(columns=["price"])
y = data["price"]

#%%
from sklearn.preprocessing import OrdinalEncoder

cut_cat = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_cat = ["J", "I", "H", "G", "F", "E", "D"]
clarity_cat = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

oe = OrdinalEncoder(categories=[cut_cat, color_cat, clarity_cat])
x[["cut", "color", "clarity"]] = oe.fit_transform(x[["cut", "color", "clarity"]])
#%%
x = np.array(x)
y = np.array(y)
#%%
from sklearn.model_selection import train_test_split

SEED = 12345

x_train, x_iter, y_train, y_iter = train_test_split(x, y, test_size=0.3, random_state=SEED)
x_val, x_test, y_val, y_test = train_test_split(x_iter, y_iter, test_size=0.5, random_state=SEED)

#%%
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_val_sc = sc.transform(x_val)
x_test_sc = sc.transform(x_test)

#%%
input_shape = x_train_sc.shape[-1]

model = tf.keras.Sequential([
    tf.keras.Input(shape = input_shape),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1)
    ])

model.summary()

model.compile(optimizer = "adam", loss="mse", metrics=["mse", "mae"])
#%%
EPOCH = 30
BATCH_SIZE = 64

es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta = 10, patience = 4)

history = model.fit(x_train_sc, y_train, validation_data = (x_val_sc, y_val), epochs = EPOCH, batch_size = BATCH_SIZE, callbacks = [es])

#%%
import matplotlib.pyplot as plt
import os

test_result = model.evaluate(x_test,y_test,batch_size=BATCH_SIZE)
print(f"Test loss = {test_result[0]}")
print(f"Test MAE = {test_result[1]}")
print(f"Test MSE = {test_result[2]}")

predictions = np.squeeze(model.predict(x_test))
labels = np.squeeze(y_test)
plt.plot(predictions,labels,".")
plt.xlabel("Predictions")
plt.ylabel("Labels")
plt.title("Graph of Predictions vs Labels with Test Data")
save_path = r"C:\Users\owner\Desktop\SHRDC MIDA AIML\Deep Learning\Git Repo\Diamond Dataset\image"
plt.savefig(os.path.join(save_path,"result.png"),bbox_inches='tight')
plt.show()
#%%