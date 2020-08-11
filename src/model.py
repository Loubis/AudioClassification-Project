from datetime import datetime

import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Activation,
    Input,
    ZeroPadding2D,
    Conv2D,
    MaxPool2D,
    Bidirectional,
    GRU,
    LSTM,
    Flatten,
    Lambda,
    concatenate,
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# import matplotlib.pyplot as plt

# Params
dict_genres = {
    "Electronic": 0,
    "Experimental": 1,
    "Folk": 2,
    "Hip-Hop": 3,
    "Instrumental": 4,
    "International": 5,
    "Pop": 6,
    "Rock": 7,
}

reverse_map = {v: k for k, v in dict_genres.items()}
BATCH_SIZE = 256  # vorher 256
EPOCH_COUNT = 50  # vorher 50


# CNN Block
def create_cnn_block(Input_Layer):
    print("Creating CNN block...")

    CNN_Block = Conv2D(
        filters=16, kernel_size=[3, 3], padding="same", activation="relu"
    )(Input_Layer)
    CNN_Block = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CNN_Block)

    CNN_Block = Conv2D(
        filters=32, kernel_size=[3, 3], padding="same", activation="relu"
    )(CNN_Block)
    CNN_Block = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CNN_Block)

    CNN_Block = Conv2D(
        filters=64, kernel_size=[3, 3], padding="same", activation="relu"
    )(CNN_Block)
    CNN_Block = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CNN_Block)

    CNN_Block = Conv2D(
        filters=128, kernel_size=[3, 3], padding="same", activation="relu"
    )(CNN_Block)
    CNN_Block = MaxPool2D(pool_size=(4, 4), strides=(4, 4))(CNN_Block)

    CNN_Block = Conv2D(
        filters=64, kernel_size=[3, 3], padding="same", activation="relu"
    )(CNN_Block)
    CNN_Block = MaxPool2D(pool_size=(4, 4), strides=(4, 4))(CNN_Block)

    CNN_Block = Flatten()(CNN_Block)

    return CNN_Block


# Bi-RNN Block
def create_birnn_block(Input_Layer):
    print("Creating BiRNN block...")

    BiRNN_Block = MaxPool2D(pool_size=(2, 1), strides=(2, 1))(Input_Layer)
    BiRNN_Block = Lambda(lambda x: tf.keras.backend.squeeze(x, axis=-1))(BiRNN_Block)
    BiRNN_Block = Bidirectional(LSTM(128))(BiRNN_Block)

    return BiRNN_Block


# Classification Block
def create_classification_block(CNN_Block, BiRNN_Block):
    print("Concatenate layers...")
    Classification_Block = concatenate([CNN_Block, BiRNN_Block], axis=-1, name="concat")

    print("Creating Classification block...")
    Output_Layer = Dense(8, activation="softmax")(Classification_Block)

    return Output_Layer


def create_parallel_cnn_birnn_model():
    Input_Layer = Input((512, 128, 1))
    CNN_Block = create_cnn_block(Input_Layer)
    BiRNN_Block = create_birnn_block(Input_Layer)
    Output_Layer = create_classification_block(CNN_Block, BiRNN_Block)

    model = Model(Input_Layer, Output_Layer)

    opt = RMSprop(lr=0.0005)  # Optimizer
    print("Compiling Model...")
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )

    model.summary()
    # tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True)

    return model


def train_model(x_train, y_train, x_val, y_val, X_test, y_test):
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)

    print("Creating model...")
    model = create_parallel_cnn_birnn_model()

    tb_callback = TensorBoard(
        log_dir="./logs/" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S"),
        histogram_freq=1,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
    )

    checkpoint_callback = ModelCheckpoint(
        "./models/parallel/weights.best.h5",
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max",
    )

    reducelr_callback = ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=10, min_delta=0.01, verbose=1
    )
    callbacks_list = [checkpoint_callback, reducelr_callback, tb_callback]

    # Fit the model and get training history.
    print("Training...")
    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCH_COUNT,
        validation_data=(x_val, y_val),
        verbose=1,
        callbacks=callbacks_list,
    )

    print("Evaluation...")
    results = model.evaluate(x_val, y_val)
    print("Evaluation results: ", results, "\n")

    # save model
    model.save("trainedModels/modelv1")

    return model, history


def run():
    print("Running model...")

    # load numpy arrays
    print("Loading train, valid and test data")

    npzfile = np.load("shuffled_train.npz")
    print(npzfile.files)
    X_train = npzfile["arr_0"]
    y_train = npzfile["arr_1"]
    print(X_train.shape, y_train.shape)

    npzfile = np.load("shuffled_valid.npz")
    print(npzfile.files)
    X_valid = npzfile["arr_0"]
    y_valid = npzfile["arr_1"]
    print(X_valid.shape, y_valid.shape)

    npzfile = np.load("test_arr.npz")
    print(npzfile.files)
    X_test = npzfile["arr_0"]
    y_test = npzfile["arr_1"]
    print(X_valid.shape, y_valid.shape)

    # start training
    model, history = train_model(X_train, y_train, X_valid, y_valid, X_test, y_test)

    # show_summary_stats(history)


if __name__ == "__main__":
    run()
