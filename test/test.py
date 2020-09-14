import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from pydicom import dcmread


img_dir = '../../prostate_images/PROSTATEx'
csv_dir = '../lib/ADC.csv'
resample = (384, 384, 19)


def main():
    prev_dcm()


# Preview roi data
def prev_data():
    train_x = np.load('../src/x_sample.npy')
    train_y = np.load('../src/y_sample.npy')
    print(train_y)

    for i in range(len(train_x)):
        print(np.shape(train_x[i]))
        print(train_y[i])

        img = train_x[i]
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 0, 2)

        fig = plt.figure()

        for num, slice in enumerate(img):
            print(np.shape(slice))
            y = fig.add_subplot(2, 2, num+1)
            new_img = slice
            y.imshow(new_img)
        plt.show()

        break


# Preview DICOM files
def prev_dcm():

    files = {}

    for dirpath, dirnames, filenames in os.walk(img_dir):
        if 'ADC' in dirpath:
            files[dirpath[29:43]] = dirpath
    
    for key in files:
        slices = [dcmread(files[key] + '/' + s).pixel_array for s in os.listdir(files[key])]
        print(np.shape(slices))

        fig = plt.figure()

        for num, slice in enumerate(slices[:12]):
            y = fig.add_subplot(3, 4, num+1)
            new_img = slice
            y.imshow(new_img)
        plt.show()

        break


# Preview NifTi files
def prev_nifti():

    data = pd.read_csv(csv_dir)
    
    for index, row in data.iterrows():

        img = nib.load('../res/converted_nifti/' + row['ProxID'] + '.nii')
        a = np.array(img.dataobj)
        a = np.swapaxes(a, 0, 2)
        print(np.shape(a))
        
        fig = plt.figure()
        
        for num, slice in enumerate(a[:12]):
            y = fig.add_subplot(3, 4, num+1)
            new_img = slice
            y.imshow(new_img)

        plt.show()
        break


if __name__ == '__main__':
    main()
