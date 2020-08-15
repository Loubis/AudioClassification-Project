import os
import sys
import traceback
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
DATA_SET_AUDIO_FILES_LARGE_PATH = DATA_SET_PATH + "/fma_large"
DATA_SET_META_DATA_PATH = DATA_SET_PATH + "/fma_metadata/tracks.csv"


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
    # Genre Dictionairy initialisieren
    dict_genres = {}

    # CSV Datei einlesen
    print("Loading meta data from fma_metadata/tracks.csv...")
    tracks_csv = pd.read_csv(DATA_SET_META_DATA_PATH, index_col=0, header=[0, 1])

    # CSV Datei auf "keep-cols" reduzieren
    print("Removing uneeded data...")
    tracks_meta = tracks_csv[keep_cols]
    if data_set == "small":
        tracks_meta = tracks_meta[tracks_meta[("set", "subset")] == "small"]
    elif data_set == "medium":
        tracks_meta = tracks_meta[tracks_meta[("set", "subset")] == "medium"]

    tracks_meta["track_id"] = tracks_meta.index

    # Daten formatieren
    print("Data format:")
    print(tracks_meta.shape, "\n")

    # Genres aller Tracks in Dictionairy speichern
    genres = tracks_meta[("track", "genre_top")].dropna().unique()
    for val in genres:
        dict_genres.update({val: len(dict_genres)})
    print("Containing classes:")
    print(dict_genres)

    # Meta Daten und Genre Dictionairy zurückgeben
    return tracks_meta, dict_genres


def create_spectogram(track_id):
    ### Load file
    filename = get_audio_path(DATA_SET_AUDIO_FILES_LARGE_PATH, track_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(filename)

    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)

    if spect.shape[1] < 512:
        return None

    # Normalize
    spect = spect[:, :512]
    spect = librosa.core.db_to_power(spect, ref=1.0)
    spect = np.log(spect)

    return spect


def create_entry(data):
    (index, row) = data
    try:
        track_id = int(row["track_id"])
        genre = str(row[("track", "genre_top")])
        if genre == 'nan':
            return None
        
        spect = create_spectogram(track_id)
    
        return spect, dict_genres[genre]
    except KeyboardInterrupt:
        exit(1)
    except Exception as e:
        print()
        print("Couldn't process create_entry: ", str(track_id))
        print(e)
        traceback.print_exc()


def create_array(df):
    genres = []
    x_spect = np.empty((0, 128, 512))
    pool = multiprocessing.Pool(8)
    for data in tqdm.tqdm(
        pool.imap_unordered(create_entry, df.iterrows()), total=df.shape[0], ncols=100
    ):
        try:
            if data is not None and data[0] is not None and data[1] is not None:
                spect, genre = data
                x_spect = np.append(x_spect, [spect], axis=0)
                genres.append(genre)

        except Exception as e:
            print()
            print(e)
            traceback.print_exc()

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
    tracks_meta, genres = get_meta_data(data_set)
    global dict_genres 
    dict_genres = genres
    
    print("Splitting into training, validation and test set")
    train = tracks_meta[tracks_meta[("set", "split")] == "training"]
    valid = tracks_meta[tracks_meta[("set", "split")] == "validation"]
    test = tracks_meta[tracks_meta[("set", "split")] == "test"]
    print(train.shape, valid.shape, test.shape, "\n")
    
    
    print("Creating test data...")
    testChunks = splitDataFrameIntoSmaller(test)
    count = 0
    for chunk in testChunks:
        count += 1
        print("Creating test data for chunk " + str(count) + "...")
        x_test, y_test = create_array(chunk)
        print("Saving test data to test_arr_" + str(count) + ".npz", "\n")
        np.savez(DATA_SET_PATH + "/pre_processed_full/test_arr_" + str(count), x_test, y_test)

    print("Creating validation data...")
    validChunks = splitDataFrameIntoSmaller(valid)
    count = 0
    for chunk in validChunks:
        count += 1
        print("Creating valid data for chunk " + str(count) + "...")
        x_valid, y_valid = create_array(chunk)
        print("Saving test data to valid_arr_" + str(count) + ".npz", "\n")
        np.savez(DATA_SET_PATH + "/pre_processed_full/valid_arr_" + str(count), x_valid, y_valid)

    print("Creating train data...")
    trainChunks = splitDataFrameIntoSmaller(train)
    count = 0
    for chunk in trainChunks:
        count += 1
        print("Creating train data for chunk " + str(count) + "...")
        x_train, y_train = create_array(chunk)
        print("Saving test data to train_arr_" + str(count) + ".npz", "\n")
        np.savez(DATA_SET_PATH + "/pre_processed_full/train_arr_" + str(count), x_train, y_train)

    print("Finishd preprocessing", "\n")


preprocess_data("full")

