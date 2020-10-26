from keras.models import Sequential
from keras.layers import Conv3D, Dense, Flatten, MaxPooling3D, Dropout
from keras.optimizers import Adam, SGD


def cnn_3d():
    model = Sequential()
    model.add(Conv3D(input_shape=(40, 40, 4, 1), filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(rate=0.4))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dropout(rate=0.4))
    model.add(Dense(units=5, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=['accuracy'])

    return model


def main():
    model = cnn_3d()
    model.summary()
    model.save('base_32')


if __name__ == '__main__':
    main()