from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau


import efficientnet.keras as efn

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


def train_model(x_train, y_train, x_val, y_val, X_test, y_test):
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)

    print("Creating model...")
    model = efn.EfficientNetB7(weights="imagenet")

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
    model.save("trainedModels/effi_net")

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
