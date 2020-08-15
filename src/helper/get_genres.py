import os
import sys
import warnings
import multiprocessing
import glob
import tqdm
import librosa
import pandas as pd
import numpy as np
from functools import partial

# Pfade
DATA_SET_PATH = "/datashare/Audio/FMA/data"
DATA_SET_AUDIO_FILES_SMALL_PATH = DATA_SET_PATH + "/fma_small"
DATA_SET_META_DATA_PATH = DATA_SET_PATH + "/fma_metadata/tracks.csv"

# zu behaltende Spalten der CSV Datei
keep_cols = [("set", "split"), ("set", "subset"), ("track", "genre_top")]


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

    print(tracks_meta[("track", "genre_top")].value_counts())

    # Genres aller Tracks in Dictionairy speichern
    genres = tracks_meta[("track", "genre_top")].unique()
    for val in genres:
        dict_genres.update({val: len(dict_genres)})
    print("Containing classes:")
    print(dict_genres)

    # Meta Daten und Genre Dictionairy zurückgeben
    return tracks_meta, dict_genres

get_meta_data("large")