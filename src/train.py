import numpy as np

from imblearn.over_sampling import RandomOverSampler
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.model_selection import train_test_split
from torchio.transforms import OneOf, RandomAffine, RandomElasticDeformation, RandomFlip, RandomNoise


# Load dataset
x_t2w = np.load('x_t2tsetra.npy')
y_t2w = np.load('y_t2tsetra.npy')

x_adc = np.load('x_adc.npy')
y_adc = np.load('y_adc.npy')

x_bval = np.load('x_bval.npy')
y_bval = np.load('y_bval.npy')

x_train = np.concatenate([x_t2w, x_adc, x_bval])
y_train = np.concatenate([y_t2w, y_adc, y_bval])

# Quadruple dataset
for i in range(2):
    x_train = np.concatenate([x_train, x_train])
    y_train = np.concatenate([y_train, y_train])

# Reshape x_train to fit oversampler
orig_shape = np.shape(x_train)
x_train = np.reshape(x_train, (orig_shape[0], orig_shape[1] * orig_shape[2] * orig_shape[3]))

# Oversample imbalanced classes so that all classes have equal occurrences
oversample = RandomOverSampler(sampling_strategy='not majority')

# Reshape x_train back to its original form
x_train, y_train = oversample.fit_resample(x_train, y_train)
x_train = np.reshape(x_train, (len(x_train), orig_shape[1], orig_shape[2], orig_shape[3]))

# Reshape and adjust dataset to prepare for training
x_train = x_train.reshape(len(x_train), 40, 40, 4, 1)
y_train = [x - 1 for x in y_train]
y_train = to_categorical(y_train, 5)

# Data augmentations to be used
transforms_dict = {
    RandomFlip(): 0.25,
    RandomElasticDeformation(): 0.25,
    RandomAffine(): 0.25,
    RandomNoise(): 0.25
}

# Create transform, with a p chance to apply augmentation
transform = OneOf(transforms_dict, p=0.8)

# Randomly augment all training data
for i in range(len(x_train)):

    # Get ROI and adjust axes for augmentation
    roi = x_train[i]
    roi = np.swapaxes(roi, 2, 3)
    roi = np.swapaxes(roi, 1, 2)
    roi = np.swapaxes(roi, 0, 1)

    # Perform augmentation, adjust axes back, and update x_train
    result = transform(roi)
    result = np.swapaxes(result, 0, 1)
    result = np.swapaxes(result, 1, 2)
    result = np.swapaxes(result, 2, 3)
    x_train[i] = result

# Split train and test data
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train)

# Train model
model = load_model('trained')
model.fit(x_train, y_train,
    epochs=10,
    validation_split=0.1,
    validation_data=(x_val, y_val),
    batch_size=1)
# model.save('trained')

# Predict for test data
predictions = model.predict(x_test)

# Print out predictions
print('\n', predictions, '\n')
correct = 0
for i in range(len(y_test)):
    print('Test Data ', i, ': Prediction - ', np.argmax(predictions[i]), ' Correct: ', np.argmax(y_test[i]))
    if np.argmax(predictions[i]) == np.argmax(y_test[i]):
        correct += 1
print('\nTest Accuracy: ', correct/len(predictions))
