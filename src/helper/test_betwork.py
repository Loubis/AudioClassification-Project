import glob
import tqdm
from yaspin import yaspin

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

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

def run():
    print("Running model...")
    # load numpy arrays
    print("Loading valid and test data")

    x_valid, y_valid, x_test, y_test = [], [], [], []

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

    print("Number of valid data: " + str(x_valid.shape) + " " + str(y_valid.shape))
    print("Number of test data: " + str(x_test.shape) + " " + str(y_test.shape))

    # load exported model
    model = keras.models.load_model('./tests/final_netz/weights.best.h5')
    
    print("Evaluation...")
    print("Validation set")
    model.evaluate(x_valid, y_valid)

    print("Test set")
    y_test_pred = model.predict(x_test)
    y_test_pred = np.argmax(y_test_pred, axis=1)
    target_names = dict_genres.keys()
    print(classification_report(y_test, y_test_pred, target_names=target_names))



if __name__ == "__main__":
    run()
