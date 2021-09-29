import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from autoprototype.tfkerasopt import KerasHPO

labels = pd.read_csv('dog/labels.csv')
print(labels.head())
train_file_location = 'dog/train/'
train_data = labels.assign(img_path = lambda x : train_file_location + x['id'] + '.jpg')
print(train_data.head())


def image_preocessing(path):
    image = Image.open(path)
    size = (128, 128)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # image= image.transpose(Image.FLIP_LEFT_RIGHT)
    image_array = np.asarray(image)
    a, b, c = (np.shape(image_array))
    if c == 4:
        image_array = image_array[:, :, :3]
    image_array = (image_array.astype(np.float32) / 255.0)

    return image_array

dir='dog/train'
prevfolder='none'
label=0
dogs = []
label_names=[]

for i in range(0,len(train_data)):
    label=train_data.iloc[i,1]
    filename=train_data.iloc[i,2]
    #print(filename)
    image = image_preocessing(filename)
    dogs.append(image)
    label_names.append(label)

print(labels)
print(len(labels),"images")


print(np.shape(label_names))
print(np.shape(dogs))

classes = len(np.unique(label_names))

Y = pd.get_dummies(label_names)



X_train=np.array(dogs)
Y_train=Y.to_numpy()



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test  = train_test_split(X_train,Y_train,test_size = 0.25)



print(X_train.shape,Y_train.shape)




train_datagen = ImageDataGenerator(shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
train_datagen.fit(X_train)

train_generator = train_datagen.flow(X_train,Y_train,batch_size = 32)

print("This is train gen", train_generator)


hpo = KerasHPO(train_generator.x,train_generator.y,EPOCHS=10,classes=120,
               max_units_fcl=400, max_conv_filters=1000,
               arch="cnn",input_shape=(128,128,3),steps_per_epoch=10)
trial , params , val_loss = hpo.get_best_parameters(n_trials=5)


