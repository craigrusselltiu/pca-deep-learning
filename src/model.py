from keras.models import Sequential
from keras.layers import Conv3D, Dense, Flatten, MaxPooling3D, Dropout
from keras.optimizers import Adam, SGD
from config import Config

# Initialise config
config = Config()


def cnn_3d():
    '''A 3D CNN model implemented using keras.
    
    Returns: A keras model
    '''

    model = Sequential()
    model.add(Conv3D(input_shape=(config.roi_x, config.roi_y, config.roi_z, 1), filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))

    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))

    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=0.4))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(rate=0.4))
    model.add(Dense(units=5, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=['accuracy'])

    return model


def main():
    '''Test model creation and show model summary, then saves model to file.
    '''

    model = cnn_3d()
    model.summary()
    model.save('models/new_base_64')


if __name__ == '__main__':
    main()