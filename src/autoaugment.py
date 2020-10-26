import numpy as np
import matplotlib.pyplot as plt

from torchio.transforms import OneOf, Compose, RandomAffine, RandomElasticDeformation, RandomFlip, RandomNoise, RandomBlur


class Policy(object):

    def __init__(self, transform1, m1, p1, transform2, m2, p2):
        ranges = {
            "flip": np.zeros(10),
            "affine": np.linspace(0, 180, 10),
            "noise" : np.linspace(0, 0.5, 10),
            "blur": np.arange(10),
            "elasticD": np.zeros(10)
        }

        transforms = {
            "flip": lambda magnitude, p: RandomFlip(p=p),
            "affine": lambda magnitude, p: RandomAffine(degrees=(magnitude), p=p),
            "noise" : lambda magnitude, p: RandomNoise(std=magnitude, p=p),
            "blur": lambda magnitude, p: RandomBlur(std=magnitude, p=p),
            "elasticD": lambda magnitude, p: RandomElasticDeformation(p=p)
        }

        self.p1 = p1
        self.m1 = ranges[transform1][m1]
        self.transform1 = transforms[transform1]
        
        self.p2 = p2
        self.m2 = ranges[transform2][m2]
        self.transform2 = transforms[transform2]


    def __call__(self, img):
        transform1 = self.transform1(self.m1, self.p2)
        transform2 = self.transform2(self.m2, self.p2)
        transform = Compose([transform1, transform2])
        return transform(img)


def preview_roi(img):
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 0, 2)

        fig = plt.figure()

        for num, each_slice in enumerate(img):
            y = fig.add_subplot(2, 2, num+1)
            new_img = each_slice
            new_img = np.reshape(new_img, (40, 40))
            y.imshow(new_img)
        plt.show()


def main():
    """
    Test and view augmentation effects on img inputs.
    """

    # Test policy transformation application
    x_train = np.load('x_adc.npy')
    x_train = np.reshape(x_train, (len(x_train), 40, 40, 4, 1))

    test_policy = Policy("flip", 0, 1, "noise", 1, 1)

    roi = x_train[0]
    preview_roi(roi)
    roi = np.swapaxes(roi, 2, 3)
    roi = np.swapaxes(roi, 1, 2)
    roi = np.swapaxes(roi, 0, 1)

    # Perform augmentation, adjust axes back, and update x_train
    result = test_policy(roi)
    result = np.swapaxes(result, 0, 1)
    result = np.swapaxes(result, 1, 2)
    result = np.swapaxes(result, 2, 3)
    preview_roi(result)


if __name__ == '__main__':
    main()