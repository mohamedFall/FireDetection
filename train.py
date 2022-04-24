import os  # operating system
import numpy as np  # linear algebra
from PIL import Image  # Python image library
import matplotlib.pyplot as plt  # making plots
# %matplotlib inline
from IPython.display import display  # displaying ?

import glob

import warnings  # ignoring unnecessary python warnings

from data_preparation import prepare_dataset

warnings.filterwarnings('ignore')

from tensorflow import keras
from keras.applications.vgg16 import VGG16  # pretrained CNN
from keras.callbacks import ModelCheckpoint  # further train the saved model
from tensorflow.keras import models, layers, optimizers  # building DNN is keras
from keras.models import load_model  # load saved model
from keras.preprocessing.image import ImageDataGenerator  # preparing image data for training

prepare_dataset()

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

conv_base.trainable = False

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

model.summary()

base_dir_train = 'C:/Users/falli/PycharmProjects/FireDetecting/Dataset'
base_dir_test = 'C:/Users/falli/PycharmProjects/FireDetecting/Test_Dataset'

# train_dir = os.path.join(base_dir_train, 'train')
train_dir_fire = os.path.join(base_dir_train, 'Fire')
train_dir_no_fire = os.path.join(base_dir_train, 'NoFire')

# test_dir = os.path.join(base_dir_test, 'test')
test_dir_fire = os.path.join(base_dir_test, 'Fire')
test_dir_no_fire = os.path.join(base_dir_test, 'NoFire')

train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_data_gen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_data_gen.flow_from_directory(base_dir_train,
                                                     target_size=(150, 150),
                                                     batch_size=32,
                                                     class_mode='binary')

test_generator = test_data_gen.flow_from_directory(base_dir_test,
                                                   target_size=(150, 150),
                                                   batch_size=32,
                                                   class_mode='binary')

history = model.fit(train_generator, epochs=30,
                    validation_data=test_generator)
model.save('VGG16_lr-4.h5')

# Dictionary to extract the numbers
hist_dict = history.history

# Training and validation accuracy
training_acc = hist_dict['acc']
validation_acc = hist_dict['val_acc']

# Training and validation loss
training_loss = hist_dict['loss']
validation_loss = hist_dict['val_loss']

# Number of epochs
epoches = range(1, 1 + len(training_acc))


def plot_func(entity):
    '''
    This function produces plot to compare the performance
    between train set and validation set.
    entity can be loss of accuracy.
    '''

    plt.figure(figsize=(8, 5))
    plt.plot(epoches, eval('training_' + entity), 'r')
    plt.plot(epoches, eval('validation_' + entity), 'b')
    plt.legend(['Training ' + entity, 'Validation ' + entity])
    plt.xlabel('Epoches')
    plt.ylabel(entity)
    plt.show()


plot_func('loss')
plot_func('acc')

conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    else:
        set_trainable = False

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

history = model.fit_generator(train_generator, epochs=50,
                              validation_data=test_generator)
model.save('VGG16_fine_tuned.h5')

hist_dict = history.history

training_accuracy = hist_dict['acc']
validation_accuracy = hist_dict['val_acc']

training_loss = hist_dict['loss']
validation_loss = hist_dict['val_loss']

epoches = range(1, 1 + len(training_acc))

# Loading the saved model
model = load_model('VGG16_fine_tuned.h5')

# taking first batch from the generator
img, label = test_generator[0]

# Predicting the images from the first batch
pred = np.round(model.predict(img)).flatten()
len(img)

# Numeric to semantic labels
label_dict = {1.0: 'No fire', 0.0: 'Fire'}

# Generating collage of plots
fig = plt.figure(figsize=(10, 9))
plt.title('Classification by the model')
plt.axis('off')

for i, img_i in enumerate(img[:20]):
    ax = fig.add_subplot(4, 5, i + 1)
    plt.axis('off')
    plt.title(label_dict[pred[i]], y=-0.2)
    ax.imshow(img_i)
