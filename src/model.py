from keras.models import Sequential
from keras.layers import Conv3D, Dense, Flatten, MaxPooling3D, Dropout
from keras.optimizers import Adam, SGD


model = Sequential()
model.add(Conv3D(input_shape=(40, 40, 4, 1), filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu'))
model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

model.add(Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', activation='relu'))
model.add(Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

model.add(Flatten())
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(rate=0.4))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(rate=0.4))
model.add(Dense(units=5, activation='softmax'))

model.compile(loss='categorical_crossentropy',
    optimizer=SGD(lr=0.01, momentum=0.99),
    metrics=['accuracy'])

model.summary()
model.save('./src/model')