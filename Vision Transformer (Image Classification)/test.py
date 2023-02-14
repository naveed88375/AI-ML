import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import sys
import warnings

warnings.filterwarnings('ignore')

#Check the dataset
if sys.argv[2] == 'cifar10':
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
elif sys.argv[2] == 'cifar100':
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
elif sys.argv[2] == 'fashion_mnist':
  (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

size = sys.argv[1]

#Load the models
cnn_model = tf.keras.models.load_model(size+'_CNN.h5')
trans_model = tf.keras.models.load_model(size+'_vision_transformer')
#Models prediction
cnn_pred = np.argmax(cnn_model.predict(x_test), axis=1)
trans_pred = np.argmax(trans_model.predict(x_test), axis=1)
#Print the performance
print("CNN model performance is as given below.")
print(classification_report(y_test, cnn_pred))
cnn_perf = cnn_model.evaluate(x_test, y_test)
print("CNN model loss on test set is:", cnn_perf[0])

print("Transformer model performance is as given below.")
print(classification_report(y_test, trans_pred))
trans_perf = trans_model.evaluate(x_test, y_test)
print("Transformer model loss on test set is:", trans_perf[0])