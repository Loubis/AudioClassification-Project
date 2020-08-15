import numpy as np
import glob

PATH = "/datashare/Audio/FMA/data/pre_processed_full/"

count = 0
for np_name in glob.glob(PATH + "*.npz"):
    npzfile = np.load(np_name)

    x = npzfile["arr_0"]
    print(x.shape)
    count = count + x.shape[0]

print(count)