from os import path, makedirs
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np

CHUNK_LENGTH_MS = 30000


def create_chunks(track_src):
    # Chunks generieren
    track = AudioSegment.from_file(track_src, "wav")
    chunks = make_chunks(track, CHUNK_LENGTH_MS)

    # Chunks in NP Array umwandeln und in gemeinsamen Array abspeichern
    chunks_arr = [len(chunks)]
    for chunk in chunks:
        chunks_arr.append(np.array(chunk.get_array_of_samples()))

    # Array der Chunks in NP Array konvertieren und zurückgeben
    chunks_arr = np.array(chunks_arr)
    print("created " + str(len(chunks)) + " chunks for " + track_src)
    print(chunks_arr)

    return chunks_arr


create_chunks("/datashare/Audio/FMA/data/converted/fma_full/000/000005.wav")
