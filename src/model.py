from datetime import datetime
import os

import glob
import tqdm
from yaspin import yaspin

import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import (
    Adam
)
from tensorflow.keras.layers import (
    Dense,
    Activation,
    BatchNormalization,
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
    Dropout,
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from sklearn.metrics import classification_report

dict_genres = {
    'Hip-Hop': 0,
    'Pop': 1,
    'Rock': 2,
    'Experimental': 3,
    'Folk': 4,
    'Jazz': 5,
    'Electronic': 6,
    'Spoken': 7,
    'International': 8,
    'Soul-RnB': 9,
    'Blues': 10,
    'Country': 11,
    'Classical': 12,
    'Old-Time / Historic': 13,
    'Instrumental': 14,
    'Easy Listening': 15
}


DATA_PATH = "/datashare/Audio/FMA/data/pre_processed_full/"

BATCH_SIZE = 64  # vorher 256
EPOCH_COUNT = 70  # vorher 50


# CNN Block
def create_cnn_block(Input_Layer):
    print("Creating CNN block...")

    CNN_Block = Conv2D(
        filters=16,
        kernel_size=[3, 3],
        padding="same",
    )(Input_Layer)
    CNN_Block = BatchNormalization()(CNN_Block)
    CNN_Block = Activation('relu')(CNN_Block)
    CNN_Block = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CNN_Block)
    CNN_Block = Dropout(0.2)(CNN_Block)

    CNN_Block = Conv2D(
        filters=32,
        kernel_size=[3, 3],
        padding="same",
    )(CNN_Block)
    CNN_Block = BatchNormalization()(CNN_Block)
    CNN_Block = Activation('relu')(CNN_Block)
    CNN_Block = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CNN_Block)
    CNN_Block = Dropout(0.2)(CNN_Block)


    CNN_Block = Conv2D(
        filters=64,
        kernel_size=[3, 3],
        padding="same",
    )(CNN_Block)
    CNN_Block = BatchNormalization()(CNN_Block)
    CNN_Block = Activation('relu')(CNN_Block)
    CNN_Block = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CNN_Block)
    CNN_Block = Dropout(0.2)(CNN_Block)


    CNN_Block = Conv2D(
        filters=128,
        kernel_size=[3, 3],
        padding="same",
    )(CNN_Block)
    CNN_Block = BatchNormalization()(CNN_Block)
    CNN_Block = Activation('relu')(CNN_Block)
    CNN_Block = MaxPool2D(pool_size=(4, 4), strides=(4, 4))(CNN_Block)
    CNN_Block = Dropout(0.2)(CNN_Block)


    CNN_Block = Conv2D(
        filters=64,
        kernel_size=[3, 3],
        padding="same",
    )(CNN_Block)
    CNN_Block = (BatchNormalization())(CNN_Block)
    CNN_Block = (Activation('relu'))(CNN_Block)
    CNN_Block = MaxPool2D(pool_size=(4, 4), strides=(4, 4))(CNN_Block)
    CNN_Block = Dropout(0.2)(CNN_Block)


    CNN_Block = Flatten()(CNN_Block)

    return CNN_Block


# Bi-RNN Block
def create_birnn_block(Input_Layer):
    print("Creating BiRNN block...")

    BiRNN_Block = MaxPool2D(pool_size=(1, 4), strides=(1, 4))(Input_Layer)

    BiRNN_Block = Lambda(lambda x: tf.keras.backend.squeeze(x, axis=-1))(BiRNN_Block)
    BiRNN_Block = Bidirectional(GRU(128))(BiRNN_Block)
    BiRNN_Block = Dropout(0.5)(BiRNN_Block)


    return BiRNN_Block


# Classification Block
def create_classification_block(CNN_Block, BiRNN_Block):
    print("Concatenate layers...")
    Classification_Block = concatenate([CNN_Block, BiRNN_Block], axis=-1, name="concat")
    Classification_Block = Dropout(0.5)(Classification_Block)

    print("Creating Classification block...")
    Output_Layer = Dense(16, activation="softmax")(
        Classification_Block
    )

    return Output_Layer


def create_parallel_cnn_birnn_model():
    print("Creating model...")
    Input_Layer = Input((128, 512, 1))
    CNN_Block = create_cnn_block(Input_Layer)
    BiRNN_Block = create_birnn_block(Input_Layer)
    Output_Layer = create_classification_block(CNN_Block, BiRNN_Block)

    model = Model(Input_Layer, Output_Layer)

    opt = Adam(learning_rate=0.002)
    print("Compiling Model...")
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )

    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True)

    return model


def train_model(model, x_train, y_train, x_valid, y_valid, x_test, y_test):
    print("Prepare training...")
    log_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    tb_callback = TensorBoard(
        log_dir="./logs/" + log_time + "/log",
        histogram_freq=1,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
    )

    checkpoint_callback = ModelCheckpoint(
        "./logs/" + log_time + "/weights.best.h5",
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
        validation_data=(x_valid, y_valid),
        verbose=1,
        callbacks=callbacks_list,
    )

    print("Evaluation...")
    print("Validation set")
    model.evaluate(x_valid, y_valid)

    print("Test set")
    y_test_pred = model.predict(x_test)
    y_test_pred = np.argmax(y_test_pred, axis=1)
    target_names = dict_genres.keys()
    print(classification_report(y_test, y_test_pred, target_names=target_names))

    # save model
    os.mkdir("./logs/" + log_time + "trained_model")
    model.save("./logs/" + log_time + "trained_model")


def load_datasets():
    print("Loading train, valid and test data")
    x_train, y_train, x_valid, y_valid, x_test, y_test = [], [], [], [], [], []

    for np_name in tqdm.tqdm(glob.glob(DATA_PATH + "train_arr_*.npz"), ncols=100):
        npzfile = np.load(np_name)
        x_train.append(npzfile["arr_0"])
        y_train.append(npzfile["arr_1"])

    with yaspin(text="Concatenating tain data", color="green") as spinner:
        x_train = np.concatenate(x_train, axis=0)
        x_train = np.expand_dims(x_train, axis=-1)
        y_train = np.concatenate(y_train, axis=0)
        spinner.ok("✅ ")

    for np_name in tqdm.tqdm(glob.glob(DATA_PATH + "valid_arr_*.npz"), ncols=100):
        npzfile = np.load(np_name)
        x_valid.append(npzfile["arr_0"])
        y_valid.append(npzfile["arr_1"])

    with yaspin(text="Concatenating validation data", color="green") as spinner:
        x_valid = np.concatenate(x_valid, axis=0)
        x_valid = np.expand_dims(x_valid, axis=-1)
        y_valid = np.concatenate(y_valid, axis=0)
        spinner.ok("✅ ")

    for np_name in tqdm.tqdm(glob.glob(DATA_PATH + "test_arr_*.npz"), ncols=100):
        npzfile = np.load(np_name)
        x_test.append(npzfile["arr_0"])
        y_test.append(npzfile["arr_1"])

    with yaspin(text="Concatenating test data", color="green") as spinner:
        x_test = np.concatenate(x_test, axis=0)
        x_test = np.expand_dims(x_test, axis=-1)
        y_test = np.concatenate(y_test, axis=0)
        spinner.ok("✅ ")

    print("Number of train data: " + str(x_train.shape) + " " + str(y_train.shape))
    print("Number of valid data: " + str(x_valid.shape) + " " + str(y_valid.shape))
    print("Number of test data: " + str(x_test.shape) + " " + str(y_test.shape))

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def run():
    print("Running model.py...")
    # load numpy arrays
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_datasets()

    # Create model
    model = create_parallel_cnn_birnn_model()

    # start training
    train_model(model, x_train, y_train, x_valid, y_valid, x_test, y_test)


if __name__ == "__main__":
    run()
