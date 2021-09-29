import urllib
from tensorflow.keras.datasets import mnist
from OptunaHPO.KerasHPO import KerasHPO
import numpy as np
import tensorflow as tf

# Register a global custom opener to avoid HTTP Error 403: Forbidden when downloading MNIST.
opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)

BATCHSIZE = 128
CLASSES = 10
EPOCHS = 20
N_TRAIN_EXAMPLES = 3000
STEPS_PER_EPOCH = int(N_TRAIN_EXAMPLES / BATCHSIZE / 10)
VALIDATION_STEPS = 30
N_TRAIN_EXAMPLES = 3000
N_VALID_EXAMPLES = 1000
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 20
trials = 2

(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
x_train = x_train.reshape(60000, 784)[:N_TRAIN_EXAMPLES].astype("float32") / 255
x_valid = x_valid.reshape(10000, 784)[:N_VALID_EXAMPLES].astype("float32") / 255

# Convert class vectors to binary class matrices.
y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_valid =np.asarray(y_valid).astype('float32').reshape((-1,1))

hpo = KerasHPO(x_train,y_train,EPOCHS=10,classes=CLASSES)#,arch="cnn",input_shape=input_shape)
trial , params , val_loss = hpo.get_best_parameters(n_trials=trials)


print("\n")
print("This is the best trial",trial)
print("\n")
print("This is the best params",params)
print("\n")
print("This is the lowest val loss",val_loss)