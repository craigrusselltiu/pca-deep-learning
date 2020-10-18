import numpy as np
import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from torchio.transforms import OneOf, Compose, RandomAffine, RandomElasticDeformation, RandomFlip, RandomNoise, RandomBlur, RandomDownsample, RandomSwap


# Config (move to config file later)
train = True
load = 'base_64'
save = 'trained_base_64'
eps = 50

# Load dataset
x_t2w = np.load('x_t2tsetra.npy')
y_t2w = np.load('y_t2tsetra.npy')

x_adc = np.load('x_adc.npy')
y_adc = np.load('y_adc.npy')

x_bval = np.load('x_bval.npy')
y_bval = np.load('y_bval.npy')

x_train = np.concatenate([x_t2w, x_adc, x_bval])
y_train = np.concatenate([y_t2w, y_adc, y_bval])

# Store starting data for prediction
x_all = x_train
x_all = x_all.reshape(len(x_all), 40, 40, 4, 1)
y_all = y_train
y_all = [x - 1 for x in y_all]
y_all = to_categorical(y_all, 5)

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

# Split train and test data
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train)

# Augment data with specified transformation(s)
def augment(data, transform):
    output = data
    for i in range(len(output)):

        # Get ROI and adjust axes for augmentation
        roi = output[i]
        roi = np.swapaxes(roi, 2, 3)
        roi = np.swapaxes(roi, 1, 2)
        roi = np.swapaxes(roi, 0, 1)

        # Perform augmentation, adjust axes back, and update x_train
        result = transform(roi)
        result = np.swapaxes(result, 0, 1)
        result = np.swapaxes(result, 1, 2)
        result = np.swapaxes(result, 2, 3)
        output[i] = result
    return output

# Data augmentations to be used
transforms_dict = {
    RandomFlip(): 1,
    RandomElasticDeformation(): 1,
    RandomAffine(): 1,
    RandomNoise(): 1,
    RandomBlur(): 1
}

# Create transform, with a p chance to apply augmentation
# transform = Compose([OneOf(transforms_dict, p=0.8), OneOf(transforms_dict, p=0.8)], p=0.95)
transform = OneOf(transforms_dict, p=0.95)
x_train = augment(x_train.copy(), transform)

# Train model
model = load_model(load)
if train:
    model.fit(x_train, y_train,
        epochs=eps,
        validation_split=0.1,
        validation_data=(x_val, y_val),
        batch_size=1,
        shuffle=True)
    model.save(save)

# Predict for test data
predictions = model.predict(x_test)

# Print out predictions
print('\n', predictions, '\n')
correct = 0
y_true = []
y_pred = []
print('Entry Number | Prediction | Actual')
for i in range(len(y_test)):
    print('Entry', i, '| Prediction:', np.argmax(predictions[i]), '| Correct:', np.argmax(y_test[i]))
    y_true.append(np.argmax(y_test[i]))
    y_pred.append(np.argmax(predictions[i]))
    if np.argmax(predictions[i]) == np.argmax(y_test[i]):
        correct += 1
print('\nTest Accuracy: ', correct/len(predictions))
print('Quadratic Weighted Kappa: ', cohen_kappa_score(y_true, y_pred, weights='quadratic'))

# Predict majority
x_flip = augment(x_all.copy(), RandomFlip())
x_ed = augment(x_all.copy(), RandomElasticDeformation())
x_affine = augment(x_all.copy(), RandomAffine())
x_noise = augment(x_all.copy(), RandomNoise())
x_blur = augment(x_all.copy(), RandomBlur())

# Create list of predictions
def predList(list):
    result = []
    for i in range(len(list)):
        result.append(np.argmax(list[i]))
    return result

y_true = predList(y_all)
y_pred = predList(model.predict(x_all.copy()))
y_flip = predList(model.predict(x_flip.copy()))
y_ed = predList(model.predict(x_ed.copy()))
y_affine = predList(model.predict(x_affine.copy()))
y_noise = predList(model.predict(x_noise.copy()))
y_blur = predList(model.predict(x_blur.copy()))
y_most = []
correct = 0

print('\nEntry Number | Prediction (None, Flip, Elastic Deformation, Affine, Noise, Blur) | Actual')
for i in range(len(y_true)):
    preds = [y_pred[i], y_flip[i], y_ed[i], y_affine[i], y_noise[i], y_blur[i]]
    most = max(set(preds), key = preds.count)
    y_most.append(most)
    print('Entry', i, '| Predictions:', preds, '| Most Occuring:', most,'| Correct:', y_true[i])
    if most == y_true[i]:
        correct += 1
print('\nTest Accuracy: ', correct/len(y_true))
print('Quadratic Weighted Kappa: ', cohen_kappa_score(y_true, y_most, weights='quadratic'))
