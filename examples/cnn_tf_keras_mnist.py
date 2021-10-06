import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(f'Shape of the training data: {x_train.shape}')
print(f'Shape of the training target: {y_train.shape}')
print(f'Shape of the test data: {x_test.shape}')
print(f'Shape of the test target: {y_test.shape}')

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train/255
x_test = x_test/255

from autoprototype.tfkerasopt import KerasHPO
hpo = KerasHPO(x_train,y_train,input_shape=(28,28,1),arch="cnn",EPOCHS=2,classes=10,steps_per_epoch=10,
               batch_size=28,loss="sparse_categorical_crossentropy")
trial , params , val_loss = hpo.get_best_parameters(n_trials=5)

