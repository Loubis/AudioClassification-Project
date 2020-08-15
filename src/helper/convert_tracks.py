from os import path, makedirs
from pydub import AudioSegment
import glob

DEST = "/datashare/Audio/FMA/data/converted"
DIR = "/datashare/Audio/FMA/data/"
SUBSET = "fma_large/"


def convert_tracks():
    i = 0
    for filename in glob.iglob(DIR + SUBSET + "**/*.mp3", recursive=True):
        # files
        src = filename
        dst = DEST + filename[25:-3] + "wav"
        if not path.exists(dst):

            # Create target Directory if don't exist
            dirName = dst[:-10]
            if not path.exists(dirName):
                makedirs(dirName)

            try:
                # convert wav to mp3
                sound = AudioSegment.from_mp3(src)
                sound.export(dst, format="wav")
                print("converted: " + filename)
                i = i + 1
            except KeyboardInterrupt:
                print("Converted " + str(i) + " tracks")
                exit()
            except:
                print("FEHLER: " + dst)

    print("Converted " + str(i) + " tracks")
    print(DEST + "/" + SUBSET)
    return DEST + "/" + SUBSET


convert_tracks()
