import os
import sys
import warnings
import multiprocessing
import glob
from functools import partial

import tqdm

import pandas as pd
import numpy as np
import librosa

DATA_SET_PATH = "/datashare/Audio/FMA/data"
DATA_SET_AUDIO_FILES_SMALL_PATH = DATA_SET_PATH + "/fma_small"
DATA_SET_META_DATA_PATH = DATA_SET_PATH + "/fma_metadata/tracks.csv"


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

keep_cols = [("set", "split"), ("set", "subset"), ("track", "genre_top")]


def get_tids_from_directory(audio_dir):
    tids = []
    for _, dirnames, files in os.walk(audio_dir):
        if dirnames == []:
            tids.extend(int(file[:-4]) for file in files)
    return tids


def get_audio_path(audio_dir, track_id):
    tid_str = "{:06d}".format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + ".mp3")


def get_meta_data(data_set):
    print("Loading meta data from fma_metadata/tracks.csv...")
    tracks_csv = pd.read_csv(DATA_SET_META_DATA_PATH, index_col=0, header=[0, 1])

    print("Removing uneeded data...")
    tracks_meta = tracks_csv[keep_cols]
    tracks_meta = tracks_meta[tracks_meta[("set", "subset")] == data_set]
    tracks_meta["track_id"] = tracks_meta.index

    print("Data format:")
    print(tracks_meta.shape, "\n")

    print("Containing classes:")
    print(tracks_meta[("track", "genre_top")].unique(), "\n")

    return tracks_meta


def create_spectogram(track_id):
    filename = get_audio_path(DATA_SET_AUDIO_FILES_SMALL_PATH, track_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(filename)
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)
    return spect.T


def create_entry(data):
    (index, row) = data
    try:
        track_id = int(row["track_id"])
        genre = str(row[("track", "genre_top")])
        spect = create_spectogram(track_id)

        # Normalize for small shape differences
        spect = spect[:512, :]

        return spect, dict_genres[genre]

    except Exception as e:
        print(e)
        print("Couldn't process create_entry: ", index)


def create_array(df):
    genres = []
    x_spect = np.empty((0, 512, 128))

    pool = multiprocessing.Pool(6)

    for data in tqdm.tqdm(
        pool.imap_unordered(create_entry, df.iterrows()), total=df.shape[0], ncols=100
    ):
        try:
            spect, genre = data
            x_spect = np.append(x_spect, [spect], axis=0)
            genres.append(genre)
        except:
            print("Couldn't process create_array: ", x_spect.shape[0] + 1)

    y_arr = np.array(genres)

    return x_spect, y_arr


def splitDataFrameIntoSmaller(df, chunkSize=1600):
    print("Splitting data into chunks of size " + str(chunkSize) + "...")
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    print("Number of chunks " + str(numberChunks), "\n")
    for i in range(numberChunks):
        listOfDf.append(df[i * chunkSize : (i + 1) * chunkSize])
    return listOfDf


def unison_shuffled_copies(a, b):
    print(len(a) == len(b), len(a), len(b))

    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def preprocess_data(data_set):
    tracks_meta = get_meta_data(data_set)

    print("Splitting into training, validation and test set")
    train = tracks_meta[tracks_meta[("set", "split")] == "training"]
    valid = tracks_meta[tracks_meta[("set", "split")] == "validation"]
    test = tracks_meta[tracks_meta[("set", "split")] == "test"]
    print(train.shape, valid.shape, test.shape, "\n")

    print("Creating test data...")
    x_test, y_test = create_array(test)
    print("Saving test data to test_arr.npz", "\n")
    np.savez("test_arr", x_test, y_test)

    print("Creating validation data...")
    x_valid, y_valid = create_array(valid)
    print("Saving test data to valid_arr.npz", "\n")
    np.savez("valid_arr", x_valid, y_valid)

    print("Creating train data...")
    dataChunks = splitDataFrameIntoSmaller(train)

    count = 0
    for chunk in dataChunks:
        count += 1
        print("Creating train data for chunk " + str(count) + "...")
        x_train, y_train = create_array(chunk)
        print("Saving test data to train" + str(count) + "_arr.npz", "\n")
        np.savez("train" + str(count) + "_arr", x_train, y_train)

    ### TODO: dynamic
    npzfile = np.load("train1_arr.npz")
    print(npzfile.files)
    X_train1 = npzfile["arr_0"]
    y_train1 = npzfile["arr_1"]
    print(X_train1.shape, y_train1.shape)

    npzfile = np.load("train2_arr.npz")
    print(npzfile.files)
    X_train2 = npzfile["arr_0"]
    y_train2 = npzfile["arr_1"]
    print(X_train2.shape, y_train2.shape)

    npzfile = np.load("train3_arr.npz")
    print(npzfile.files)
    X_train3 = npzfile["arr_0"]
    y_train3 = npzfile["arr_1"]
    print(X_train3.shape, y_train3.shape)

    npzfile = np.load("train4_arr.npz")
    print(npzfile.files)
    X_train4 = npzfile["arr_0"]
    y_train4 = npzfile["arr_1"]
    print(X_train4.shape, y_train4.shape)

    npzfile = np.load("valid_arr.npz")
    print(npzfile.files)
    X_valid = npzfile["arr_0"]
    y_valid = npzfile["arr_1"]
    print(X_valid.shape, y_valid.shape)

    npzfile = np.load("test_arr.npz")
    print(npzfile.files)
    X_test = npzfile["arr_0"]
    y_test = npzfile["arr_1"]
    print(X_test.shape, y_test.shape)

    X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4), axis=0)
    y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4), axis=0)
    print(X_train.shape, y_train.shape)

    ## Convert y data from scale 0-7

    ### Convert the scale of training data
    X_train_raw = librosa.core.db_to_power(X_train, ref=1.0)
    print(np.amin(X_train_raw), np.amax(X_train_raw), np.mean(X_train_raw))
    X_train_log = np.log(X_train_raw)
    print(np.amin(X_train_log), np.amax(X_train_log), np.mean(X_train_log))

    X_valid_raw = librosa.core.db_to_power(X_valid, ref=1.0)
    X_valid_log = np.log(X_valid_raw)

    X_test_raw = librosa.core.db_to_power(X_test, ref=1.0)
    X_test_log = np.log(X_test_raw)

    X_train, y_train = unison_shuffled_copies(X_train_log, y_train)
    X_valid, y_valid = unison_shuffled_copies(X_valid_log, y_valid)
    X_test, y_test = unison_shuffled_copies(X_test_log, y_test)

    print("converting")

    print("Shapes are: ", X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

    np.savez("shuffled_train", X_train, y_train)
    np.savez("shuffled_valid", X_valid, y_valid)
    np.savez("shuffled_test", X_test, y_test)

    print("Finishd preprocessing", "\n")


preprocess_data("small")

