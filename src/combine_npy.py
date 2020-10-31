import numpy as np


def main():
    x_t2w = np.load('data/x_t2tsetra.npy')
    y_t2w = np.load('data/y_t2tsetra.npy')

    x_adc = np.load('data/x_adc.npy')
    y_adc = np.load('data/y_adc.npy')

    x_bval = np.load('data/x_bval.npy')
    y_bval = np.load('data/y_bval.npy')

    x_train = np.concatenate([x_t2w, x_adc, x_bval])
    y_train = np.concatenate([y_t2w, y_adc, y_bval])

    np.save('data/x_train', x_train)
    np.save('data/y_train', y_train)


if __name__ == '__main__':
    main()