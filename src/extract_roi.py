import numpy as np
import os
import pandas as pd
import re

from config import Config
from operator import mul
from pydicom import dcmread
from scipy.ndimage import zoom

# Initialise config class
config = Config()


# Load data
data = pd.read_csv('../lib/BVAL.csv')

imgs = {}
for dirpath, dirnames, filenames in os.walk(config.img_path):
    if 'BVAL' in dirpath:
        imgs[dirpath[32:46]] = dirpath

x = []
y = []

for index, row in data.iterrows():

    dimensions = eval(row["VoxelSpacing"])

    array = [dcmread(imgs[row["ProxID"]] + '/' + s).pixel_array for s in os.listdir(imgs[row["ProxID"]])]
    array = np.swapaxes(array, 0, 2)
    array = zoom(array, dimensions) # resample to 1x1x1 voxel dimensions
    array = (array - array.min()) / (array.max() - array.min()) # normalise 0 to 1

    position = eval(re.sub('\s+', ',', row["ijk"])) # get lesion center
    position = tuple(map(int, map(mul, dimensions, position))) # correct to resampled position 

    roi = array[
        int(position[0] - config.roi_x/2) : int(position[0] + config.roi_x/2),
        int(position[1] - config.roi_y/2) : int(position[1] + config.roi_y/2),
        int(position[2] - config.roi_z/2) : int(position[2] + config.roi_z/2)
    ] # extract ROI volume

    x.append(roi)
    y.append(row["ggg"])
    print(f'Processed image {index}')

x = np.array(x)
y = np.array(y)

np.save('data/test_x_bval', x)
np.save('data/test_y_bval', y)