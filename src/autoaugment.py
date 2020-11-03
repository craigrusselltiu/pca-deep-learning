import numpy as np
import matplotlib.pyplot as plt

from random import randint
from torchio.transforms import OneOf, Compose, RandomAffine, RandomElasticDeformation, RandomFlip, RandomNoise, RandomBlur


class Policy(object):

    def __init__(self, transform1, m1, p1, transform2, m2, p2):
        ranges = {
            'flip': np.zeros(10),
            'affine': np.linspace(0, 180, 10),
            'noise' : np.linspace(0, 0.5, 10),
            'blur': np.arange(10),
            'elasticD': np.zeros(10)
        }

        transforms = {
            'flip': lambda magnitude, p: RandomFlip(p=p),
            'affine': lambda magnitude, p: RandomAffine(degrees=(magnitude), p=p),
            'noise' : lambda magnitude, p: RandomNoise(std=magnitude, p=p),
            'blur': lambda magnitude, p: RandomBlur(std=magnitude, p=p),
            'elasticD': lambda magnitude, p: RandomElasticDeformation(p=p)
        }

        self.transform1 = transforms[transform1]
        self.t1_input = transform1
        self.m1 = ranges[transform1][m1]
        self.m1_input = m1
        self.p1 = p1
        
        self.transform2 = transforms[transform2]
        self.t2_input = transform2
        self.m2 = ranges[transform2][m2]
        self.m2_input = m2
        self.p2 = p2

        self.kappa = 0.0
        

    def __call__(self, img):
        transform1 = self.transform1(self.m1, self.p2)
        transform2 = self.transform2(self.m2, self.p2)
        transform = Compose([transform1, transform2])
        # print('Policy Selected: (\'' + self.t1_input + '\', ' + str(self.m1_input) + ', ' + str(self.p1) + ', \'' + self.t2_input + '\', ' + str(self.m2_input) + ', ' + str(self.p2) + ')')
        return transform(img)


class AlphaPolicy(object):
    
    def __init__(self):
        self.policies = [
            Policy('noise', 5, 0.5, 'flip', 6, 0.1),
            Policy('elasticD', 7, 0.5, 'flip', 9, 0.7),
            Policy('blur', 5, 0.8, 'noise', 9, 0.4),
            Policy('elasticD', 7, 0.7, 'blur', 1, 0.4),

            Policy('affine', 8, 0.7, 'elasticD', 8, 0.2),
            Policy('flip', 7, 0.9, 'blur', 4, 0.3),
            Policy('blur', 5, 1.0, 'elasticD', 6, 0.1),
            Policy('noise', 6, 0.7, 'flip', 6, 0.4),

            Policy('flip', 6, 0.4, 'affine', 9, 0.2),
            Policy('elasticD', 3, 0.9, 'flip', 8, 0.1),
            Policy('flip', 1, 0.3, 'noise', 2, 0.1),
            Policy('flip', 2, 0.6, 'elasticD', 4, 0.6),

            Policy('affine', 1, 0.9, 'elasticD', 8, 0.3),
            Policy('flip', 9, 0.3, 'noise', 1, 0.2),
            Policy('noise', 2, 0.2, 'elasticD', 2, 0.5),
            Policy('elasticD', 3, 0.5, 'flip', 3, 0.3),

            Policy('flip', 2, 1.0, 'blur', 7, 0.1),
            Policy('flip', 5, 0.5, 'noise', 2, 0.4),
            Policy('flip', 1, 0.5, 'affine', 6, 0.5),
            Policy('blur', 1, 0.2, 'flip', 5, 0.1),

            Policy('affine', 5, 0.7, 'blur', 5, 0.3),
            Policy('noise', 1, 0.7, 'flip', 6, 0.2),
            Policy('elasticD', 2, 0.3, 'flip', 9, 0.1),
            Policy('affine', 1, 0.7, 'flip', 2, 0.6)
        ]


    def __call__(self, img):
        policy = randint(0, len(self.policies) - 1)
        return self.policies[policy](img)


    def __repr__(self):
        return 'AlphaPolicy'


class BetaPolicy(object):
    
    def __init__(self):
        self.policies = [
            Policy('noise', 5, 0.5, 'flip', 6, 0.1),
            Policy('elasticD', 7, 0.5, 'flip', 9, 0.7),
            Policy('blur', 5, 0.8, 'noise', 9, 0.4),
            Policy('elasticD', 7, 0.7, 'blur', 1, 0.4),

            Policy('affine', 8, 0.7, 'elasticD', 8, 0.2),
            Policy('flip', 7, 0.9, 'blur', 4, 0.3),
            Policy('blur', 5, 1.0, 'elasticD', 6, 0.1),
            Policy('noise', 6, 0.7, 'flip', 6, 0.4),

            Policy('flip', 6, 0.4, 'affine', 9, 0.2),
            Policy('elasticD', 3, 0.9, 'flip', 8, 0.1),
            Policy('flip', 1, 0.3, 'noise', 2, 0.1),
            Policy('flip', 2, 0.6, 'elasticD', 4, 0.6)
        ]


    def __call__(self, img):
        policy = randint(0, len(self.policies) - 1)
        return self.policies[policy](img)


    def __repr__(self):
        return 'BetaPolicy'


class GammaPolicy(object):
    
    def __init__(self):
        self.policies = [
            Policy('blur', 3, 0.3, 'affine', 4, 0.0),
            Policy('affine', 6, 0.1, 'flip', 8, 0.0),
            Policy('blur', 2, 1.0, 'affine', 5, 0.0),
            Policy('affine', 6, 0.4, 'elasticD', 1, 0.0),

            Policy('flip', 6, 0.8, 'affine', 7, 0.0),
            Policy('noise', 4, 0.6, 'flip', 8, 0.0),
            Policy('blur', 3, 0.7, 'noise', 1, 0.0),
            Policy('noise', 1, 1.0, 'flip', 8, 0.8),

            Policy('flip', 4, 0.2, 'noise', 2, 0.0),
            Policy('blur', 5, 0.6, 'elasticD', 6, 0.0),
            Policy('elasticD', 7, 0.6, 'blur', 3, 0.0),
            Policy('noise', 1, 0.5, 'flip', 8, 0.3)
        ]


    def __call__(self, img):
        policy = randint(0, len(self.policies) - 1)
        return self.policies[policy](img)


    def __repr__(self):
        return 'GammaPolicy'


def preview_roi(img):
    '''Previews a 3D ROI using pyplot.

    Usage: preview_roi(image)
    '''

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
    '''Test and view augmentation effects on img inputs.
    '''

    # Test policy transformation application
    x_train = np.load('x_train.npy')
    x_train = np.reshape(x_train, (len(x_train), 40, 40, 4, 1))

    test_policy = Policy('flip', 0, 1, 'noise', 2, 1)

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