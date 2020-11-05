import numpy as np


def main():
    x_t2w = np.load('data/test_x_t2tsetra.npy')
    y_t2w = np.load('data/test_y_t2tsetra.npy')

    x_adc = np.load('data/test_x_adc.npy')
    y_adc = np.load('data/test_y_adc.npy')

    x_bval = np.load('data/test_x_bval.npy')
    y_bval = np.load('data/test_y_bval.npy')

    x_train = np.concatenate([x_t2w, x_adc, x_bval])
    y_train = np.concatenate([y_t2w, y_adc, y_bval])

    np.save('data/test_x_train', x_train)
    np.save('data/test_y_train', y_train)


if __name__ == '__main__':
    main()