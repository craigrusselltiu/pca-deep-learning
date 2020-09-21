import numpy as np
import os
import pandas as pd
import re

from operator import mul
from pydicom import dcmread
from scipy.ndimage import zoom


img_dir = '../../prostate_images/PROSTATEx'


# Load data
data = pd.read_csv('../lib/t2_tse_tra.csv')

imgs = {}
for dirpath, dirnames, filenames in os.walk(img_dir):
    if 't2tsetra' in dirpath:
        imgs[dirpath[32:46]] = dirpath

x = []
y = []

for index, row in data.iterrows():

    dimensions = eval(row["VoxelSpacing"])

    # img = nib.load("./converted_nifti/" + row["ProxID"] + ".nii") # loads image
    # array = img.get_fdata() # convert to array

    array = [dcmread(imgs[row["ProxID"]] + '/' + s).pixel_array for s in os.listdir(imgs[row["ProxID"]])]
    
    array = np.swapaxes(array, 0, 2)
    
    array = zoom(array, dimensions) # resample to 1x1x1 voxel dimensions
    array = (array - array.min()) / (array.max() - array.min()) # normalise 0 to 1

    position = eval(re.sub('\s+', ',', row["ijk"])) # get lesion center
    
    position = tuple(map(int, map(mul, dimensions, position))) # correct to resampled position  
    print(np.shape(array), position)
    roi = array[position[0]-20:position[0]+20, position[1]-20:position[1]+20, position[2]-2:position[2]+2] # extract ROI volume

    x.append(roi)
    y.append(row["ggg"])
    print(f'Processed image {index}')
    print(roi.shape)

x = np.array(x)
y = np.array(y)

np.save('x_t2tsetra', x)
np.save('y_t2tsetra', y)