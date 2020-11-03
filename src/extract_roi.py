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
data = pd.read_csv('../lib/test_t2_tse_tra.csv')

imgs = {}
for dirpath, dirnames, filenames in os.walk(config.img_path):
    if 't2tsetra' in dirpath:
        imgs[dirpath[44:58]] = dirpath

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
    roi = array[position[0]-20:position[0]+20, position[1]-20:position[1]+20, position[2]-2:position[2]+2] # extract ROI volume

    x.append(roi)
    y.append(row["ggg"])
    print(f'Processed image {index}')

x = np.array(x)
y = np.array(y)

np.save('data/test_x_t2tsetra', x)
np.save('data/test_y_t2tsetra', y)