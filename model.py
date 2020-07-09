import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Input, ZeroPadding2D, Conv2D, MaxPool2D, Bidirectional, GRU, Flatten, Lambda, concatenate


# Input Layer
Input_Layer = Input((128,513,1))

# CNN Block
CNN_Block = ZeroPadding2D(padding=(1,1))(Input_Layer)
CNN_Block = Conv2D(filters=16, kernel_size=[3,3], padding="valid", activation="relu")(CNN_Block)
CNN_Block = MaxPool2D(pool_size=(2,2), strides=(2,2))(CNN_Block)

CNN_Block = ZeroPadding2D(padding=(1,1))(CNN_Block)
CNN_Block = Conv2D(filters=32, kernel_size=[3,3], padding="valid", activation="relu")(CNN_Block)
CNN_Block = MaxPool2D(pool_size=(2,2), strides=(2,2))(CNN_Block)

CNN_Block = ZeroPadding2D(padding=(1,1))(CNN_Block)
CNN_Block = Conv2D(filters=64, kernel_size=[3,3], padding="valid", activation="relu")(CNN_Block)
CNN_Block = MaxPool2D(pool_size=(2,2), strides=(2,2))(CNN_Block)

CNN_Block = ZeroPadding2D(padding=(1,1))(CNN_Block)
CNN_Block = Conv2D(filters=128, kernel_size=[3,3], padding="valid", activation="relu")(CNN_Block)
CNN_Block = MaxPool2D(pool_size=(4,4), strides=(4,4))(CNN_Block)

CNN_Block = ZeroPadding2D(padding=(1,1))(CNN_Block)
CNN_Block = Conv2D(filters=64, kernel_size=[3,3], padding="valid", activation="relu")(CNN_Block)
CNN_Block = MaxPool2D(pool_size=(4,4), strides=(4,4))(CNN_Block)

CNN_Block = Flatten()(CNN_Block)

# Bi-RNN Block
BiRNN_Block = MaxPool2D(pool_size=(4,2), strides=(1,2))(Input_Layer)
BiRNN_Block = Lambda(lambda x: tf.keras.backend.squeeze(x, axis=-1))(BiRNN_Block)
BiRNN_Block = Bidirectional(GRU(128))(BiRNN_Block)

# Classification Block

# Merge CNN and BiRNN output Vector
Classification_Block = concatenate([CNN_Block, BiRNN_Block], axis=-1, name='concat')

Output_Layer = Dense(10, activation="softmax", input_shape=(512,))(Classification_Block)

model = Model(Input_Layer, Output_Layer)

model.summary()

# tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True)