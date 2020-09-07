import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


x_train = np.load('./src/x_train.npy')
y_train = np.load('./src/y_train.npy')

x_train = x_train / 255.0
print(np.min(x_train[0]))
x_train = x_train.reshape(len(x_train), 40, 40, 4, 1)
y_train = [x - 1 for x in y_train]
y_train = to_categorical(y_train, 5)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)

model = load_model('./src/model')
model.fit(x_train, y_train,
    epochs=50,
    validation_split=0.1,
    batch_size=len(x_train))

predictions = model.predict(x_test)
print(predictions)
for i in range(len(y_test)):
    print('Entry ', i)
    print('Prediction: ', np.argmax(predictions[i]), ' Correct: ', np.argmax(y_test[i]))