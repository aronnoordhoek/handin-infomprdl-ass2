# -*- coding: utf-8 -*-
"""prdl_assignement2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hV2X2RYg4a3vNBW8-WTptYM7z8jusjUt
"""

import os
import h5py
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
import torch

from google.colab import drive
drive.mount('/content/drive')

hdFileName = '/content/drive/MyDrive/cross.h5'
modeType   = 'r'
f = h5py.File(hdFileName, modeType)
print(f)

X_train = np.array(f['train']['data_1d'])
y_train = np.array(f['train']['labels'])

X_test = np.array(f['test2']['data_1d'])
y_test = np.array(f['test2']['labels'])

x, y, xt, yt = map(torch.tensor, (X_train, y_train, X_test, y_test))
x.shape, y.shape, xt.shape, yt.shape

x_tf = x.permute(0, 2, 1).to("cpu").detach().numpy() #x.permute(0, 2, 1).to("cpu").detach().numpy()
y_tf = y.to("cpu").detach().numpy()

x_tf_val = xt.permute(0, 2, 1).to("cpu").detach().numpy()
y_tf_val = yt.to("cpu").detach().numpy()
x_tf.shape, y_tf.shape, x_tf_val.shape, y_tf_val.shape

# Commented out IPython magic to ensure Python compatibility.
# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss") #Validation loss Accuracy
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    #plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

def plot_learning_curves_acc(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Accuracy") #Validation loss Accuracy
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    #plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

def plot_learning_curves_acc_val(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Validation loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation accuracy") #Validation loss Accuracy
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    #plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

def plot_predictions(loss, val_loss):
    plt.plot(np.arange(len(loss)), loss, "b.-", label="Actuals")
    plt.plot(np.arange(len(val_loss)), val_loss, "r.-", label="Predictions")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    #plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("no. of signals")
    plt.ylabel("signals")
    plt.grid(True)

class GatedActivationUnit(keras.layers.Layer):
    def __init__(self, activation="tanh", **kwargs): #(self, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
    def call(self, inputs):
        n_filters = inputs.shape[-1] // 2
        linear_output = self.activation(inputs[..., :n_filters])
        gate = keras.activations.sigmoid(inputs[..., n_filters:])
        return self.activation(linear_output) * gate

def wavenet_residual_block(inputs, n_filters, dilation_rate):
    z = keras.layers.Conv1D(2 * n_filters, kernel_size=2, padding="causal",
                            dilation_rate=dilation_rate)(inputs)                #padding="SAME" padding="causal"
    z = GatedActivationUnit()(z)
    z = keras.layers.Conv1D(n_filters, kernel_size=1)(z)
    return keras.layers.Add()([z, inputs]), z

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

n_layers_per_block = 10 #4 #4 # 3 # 10 in the paper
n_blocks = 3 #3 #1 # 3 in the paper
n_filters = 128 #64 #32 # 128 in the paper
n_outputs = 248 #1 #10 # 256 in the paper

inputs = keras.layers.Input(shape=[None, 248])  #[None, 3] [None,1800,32768, 3]
z = keras.layers.Conv1D(n_filters, kernel_size=2, padding="causal")(inputs)   #padding="causal" padding="SAME"

z = keras.layers.BatchNormalization()(z) #new
#z = keras.layers.MaxPool1D(pool_size=3, strides=2, padding="SAME")(z) #new

skip_to_last = []
for dilation_rate in [2**i for i in range(n_layers_per_block)] * n_blocks:
    z, skip = wavenet_residual_block(z, n_filters, dilation_rate)
    skip_to_last.append(skip)

z = keras.activations.relu(keras.layers.Add()(skip_to_last))
z = keras.layers.Conv1D(n_filters, kernel_size=1, activation="relu")(z)

s = keras.layers.Conv1D(n_outputs, kernel_size=1, activation='sigmoid')(z) #new
signals = keras.models.Model(inputs=[inputs], outputs=[s]) #new

z = keras.layers.GlobalAvgPool1D()(z) #new
z = keras.layers.Flatten()(z) #new

Y_proba = keras.layers.Dense(4, activation="softmax")(z)
#keras.layers.Conv1D(n_outputs, kernel_size=1, activation='sigmoid')(z)  #activation="softmax" 'sigmoid'

model = keras.models.Model(inputs=[inputs], outputs=[Y_proba])

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"]) #"categorical_crossentropy"
history = model.fit(x_tf, y_tf, epochs=5, batch_size=64,
                    validation_data=(x_tf_val, y_tf_val))

model.evaluate(x_tf, y_tf)

model.evaluate(x_tf_val, y_tf_val)

xx = signals.predict(x_tf)
print(xx.shape, x_tf.shape)

model_preds = model.predict(x_tf)
print(model_preds.shape, y_tf.shape)

plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

plot_learning_curves_acc(history.history["loss"], history.history["accuracy"])
plt.show()

plot_learning_curves_acc_val(history.history["val_loss"], history.history["val_accuracy"])
plt.show()

plot_predictions(y_tf[0], model_preds[0])
plt.show()

fig, axs = plt.subplots(2)
fig.suptitle('Signals')
fig.set_size_inches(8, 7, forward=True)
no = 0
axs[0].plot(xx[no]) #,'r', color='b'
axs[1].plot((x_tf[no,:,:]))
axs[0].set_title("model signals 0")
axs[1].set_title("raw signals 0")

fig.tight_layout()
plt.show

fig, axs = plt.subplots(2)
fig.suptitle('Signals')
fig.set_size_inches(8, 7, forward=True)
no = 3000
axs[0].plot(xx[no]) #,'r', color='b'
axs[1].plot((x_tf[no,:,:]))
axs[0].set_title("model signals 3000")
axs[1].set_title("raw signals 3000")

fig.tight_layout()
plt.show

eval_predictions = model.predict(x_tf_val)
xx_eval = signals.predict(x_tf_val)
print(xx_eval.shape, x_tf_val.shape)

plot_predictions(y_tf_val[800], eval_predictions[800])
plt.show()

fig, axs = plt.subplots(2)
fig.suptitle('Validation Signals')
fig.set_size_inches(8, 7, forward=True)

no = 300
axs[0].plot(xx_eval[no]) #,'r', color='b'
axs[1].plot((x_tf_val[no,:,:]))
axs[0].set_title("validation signals 300")
axs[1].set_title("raw signals 300")

fig.tight_layout()
plt.show

fig, axs = plt.subplots(8)
fig.suptitle('Signals')
fig.set_size_inches(15, 25, forward=True)

axs[0].plot(xx[2]) #,'r', color='b'
axs[1].plot((x_tf[2,:,:]))
axs[0].set_title("model signals 2")
axs[1].set_title("raw signals 2")

axs[2].plot(xx[300]) #,'r', color='g'
axs[3].plot((x_tf[300,:,:]))
axs[2].set_title("model signals 300")
axs[3].set_title("raw signals 300")

axs[4].plot(xx[800]) #,'r', color='r'
axs[5].plot((x_tf[800,:,:]))
axs[4].set_title("model signals 800")
axs[5].set_title("raw signals 800")

axs[6].plot(xx[1200]) #,'r', color='r'
axs[7].plot((x_tf[1200,:,:]))
axs[6].set_title("model signals 1200")
axs[7].set_title("raw signals 1200")


fig.tight_layout()
plt.show