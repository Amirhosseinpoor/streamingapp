import numpy as np 
import tensorflow as tf
from keras.metrics import TruePositives
import keras.backend as K
from keras import models,layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from keras.utils import to_categorical

plt.style.use('ggplot')
import time
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

data = []
labels = []
epochs = 100
name = "Corneal_thickness"
for item in glob.glob(name + "\*\*"):
    image = cv.imread(item)
    image = cv.resize(image,(180,180))
    labels.append(item.split("\\")[-2])
    data.append(image)
data = np.array(data)
labels = np.array(labels)

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)
xTrain,xTest,yTrain,yTest = train_test_split(data,labels,test_size=0.2)

net = models.Sequential([
    layers.Conv2D(filters=16,kernel_size=(3,3),activation="relu",input_shape = (180,180,3),padding="same"),
    layers.MaxPool2D(pool_size=(2,2),strides=2),
    layers.BatchNormalization(),
    layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu",padding="same"),
    layers.MaxPool2D(pool_size=(2,2),strides=2),
    layers.BatchNormalization(),
    # layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu",padding="same"),
    # layers.MaxPool2D(pool_size=(2,2),strides=2),
    # layers.BatchNormalization(),
    # layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu",padding="same"),
    # layers.MaxPool2D(pool_size=(2,2),strides=2),
    # layers.BatchNormalization(),
    # layers.Conv2D(filters=128,kernel_size=(3,3),activation="relu",padding="same"),
    # layers.MaxPool2D(pool_size=(2,2),strides=2),
    # layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(210,activation= "relu"),
    layers.Dense(2,activation="softmax")
])
net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001),loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['acc',f1_m,precision_m, recall_m])
net.summary()

history = net.fit(xTrain,yTrain,batch_size=16,epochs=epochs,validation_data=(xTest,yTest))
print(history.history)
train_acc = history.history['acc']
val_acc = history.history['val_acc']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
val_f1 = history.history['val_f1_m']
train_f1 = history.history['f1_m']
train_precision = history.history['precision_m']
train_recall = history.history['recall_m']
val_precision = history.history['val_precision_m']
val_recall = history.history['val_recall_m']

epochs = range(1,epochs + 1)
fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(5,1)
# fig.suptitle('Horizontally stacked subplots')
ax1.plot(epochs, train_acc, 'g', label='Training Accuracy')
ax1.plot(epochs, val_acc, 'b', label='validation Accuracy')
ax2.plot(epochs, train_loss, 'g', label='Training loss')
ax2.plot(epochs, val_loss, 'b', label='validation loss')
ax1.set_title('Training and Validation Accuracy')
ax2.set_title('Training and Validation Loss')
ax1.set_xlabel('Epochs')
ax2.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Loss')
ax3.plot(epochs, train_f1,'g',label = "Train F1 score")
ax3.plot(epochs, val_f1,'r',label = "Validation F1 score")
ax3.set_title('Training and Validation F1 score')
ax3.set_xlabel(epochs)
ax3.set_ylabel('F1')
ax4.plot(epochs, train_precision,'g',label = "Train  Preccision")
ax4.plot(epochs, val_precision,'r',label = "Validation  Preccision")
ax4.set_title('Training and Validation  Preccision')
ax4.set_xlabel(epochs)
ax4.set_ylabel('Preccision')
ax5.plot(epochs, train_recall,'g',label = "Train  recall")
ax5.plot(epochs, val_recall,'r',label = "Validation  recall")
ax5.set_title('Training and Validation  Recall')
ax5.set_xlabel(epochs)
ax5.set_ylabel('Recall')
# plt.title("")
plt.legend()
net.save(name+".h5")
plt.savefig(name + ".png")
plt.show()
plt.close(fig)