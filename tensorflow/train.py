from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers as opt

from helpers import backend_diou, diou_loss, CircleParams

import pandas as pd
import numpy as np
import tensorflow as tf

train_results_path = "datasets/trainset/"

model = Sequential()

# model.add(Conv2D(filters = 32, kernel_size = 5, data_format = 'channels_first', input_shape=(1, 1, 200, 200), output_shape=(32, 32, 98, 98)))
model.add(Conv2D(32, (3, 3), input_shape=(200, 200, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(filters = 64, kernel_size = 3, data_format = 'channels_first', input_shape=(32, 32, 98, 98), output_shape=(64, 64, 48, 48)))
model.add(Conv2D(64, (5, 5)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(filters = 128, kernel_size = 3, data_format = 'channels_first', input_shape=(64, 64, 48, 48), output_shape=(128, 128, 23, 23)))
model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(filters = 4, kernel_size = 1, data_format = 'channels_first', input_shape=(128, 128, 23, 23), output_shape=(128, 4, 21, 21)))
model.add(Conv2D(4, (1, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(256, input_shape=(4*21*21,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(16, input_shape=(256,)))
model.add(Activation('relu'))
model.add(Dense(3, input_shape=(16,)))


# Optimizers + Compilation
adam_opt = opt.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.005, amsgrad=False)
model.compile(loss=diou_loss,
              optimizer=adam_opt,
              # optimizer=optimizers.RMSprop(lr=.002),
              metrics=[backend_diou],
              run_eagerly=True)

# Load and reshape training data
y_temp = pd.read_csv("train_set.csv")
y_temp.drop('PATH', axis=1, inplace=True)
y_train = y_temp.to_numpy()

x_trainlist = []

for filepath in y_temp['PATH']:
    x_trainlist.append(np.load(filepath))

x_train = np.stack(x_trainlist, axis=0)

# Train model and save
model.fit(x=x_train, y=y_train, epochs=6)

print('Finished Training')

model.save('models/circle_cnn_pure_diou.keras')