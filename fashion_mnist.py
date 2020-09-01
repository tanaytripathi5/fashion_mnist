import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

train_data = pd.read_csv(r'D:\Projects\Fashion-mnist\fashion-mnist_train.csv\fashion-mnist_train.csv')
val_data =  pd.read_csv(r'D:\Projects\Fashion-mnist\fashion-mnist_test.csv\fashion-mnist_test.csv')

X_train = train_data.iloc[:,1:].values
y_train = train_data.iloc[:,0].values

X_val = val_data.iloc[:,1:].values
y_val = val_data.iloc[:,0].values

X_train,X_val,y_train,y_val = np.array(X_train),np.array(X_val),np.array(y_train),np.array(y_val)

X_train = X_train.reshape(-1,28,28,1)
X_val = X_val.reshape(-1,28,28,1)

X_train = X_train/255.0
X_val = X_val/255.0

p = X_train.shape

model = tf.keras.models.Sequential()
# Conv-Pool Layer 1
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size =(3,3),activation='relu',input_shape=(28,28,1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D())
# Conv-Pool Layer 2
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size =(3,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D())
# Conv-Pool Layer 3
model.add(tf.keras.layers.Conv2D(filters=128,kernel_size =(3,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D())
# Flatten and Affine Layers 
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(30,activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

# Compiling Model
model.compile(optimizer ='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# training the model on training_set and valing it on validation set
history = model.fit(X_train,y_train,validation_data=(X_val, y_val),epochs =20,batch_size=32)

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
