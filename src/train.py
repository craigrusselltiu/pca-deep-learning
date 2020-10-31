import numpy as np
import matplotlib.pyplot as plt

from autoaugment import Policy, AlphaPolicy
from config import Config
from imblearn.over_sampling import RandomOverSampler
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import load_model
from search import augment_imgs
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from torchio.transforms import OneOf, Compose, RandomAffine, RandomElasticDeformation, RandomFlip, RandomNoise, RandomBlur, RandomDownsample, RandomSwap

# Initialise config class
config = Config()


def preprocess(x, y):
    '''Preprocess x and y inputs for training.

    Returns: x_train, y_train, x_val, y_val, x_test, y_val
    '''

    # Quadruple dataset
    for i in range(2):
        x_train = np.concatenate([x, x])
        y_train = np.concatenate([y, y])

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

    # Split train, validate, and test data
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train)
    return x_train, y_train, x_val, y_val, x_test, y_test


def augment(data, transform):
    '''Augment input data with the provided transformation.

    Usage: augment(data, transform)
    '''

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


def random_augment(x):
    '''Randomly augment input data.

    Returns: Randomly augmented input
    '''

    # Data augmentations to be used
    transforms_dict = {
        RandomFlip(): 1,
        RandomElasticDeformation(): 1,
        RandomAffine(): 1,
        RandomNoise(): 1,
        RandomBlur(): 1
    }

    # Create random transform, with a p chance to apply augmentation
    transform = OneOf(transforms_dict, p=0.95)
    return augment(x, transform)


def predict(model, x, y):
    '''Provides model predictions from given test data.

    Usage: predict(model, x_test, y_test)
    '''

    # Predict for test data
    predictions = model.predict(x)

    # Print out predictions
    print('\n', predictions, '\n')
    correct = 0
    y_true = []
    y_pred = []
    print('Entry Number | Prediction | Actual')
    for i in range(len(y)):
        print('Entry', i, '| Prediction:', np.argmax(predictions[i]), '| Correct:', np.argmax(y[i]))
        y_true.append(np.argmax(y[i]))
        y_pred.append(np.argmax(predictions[i]))
        if np.argmax(predictions[i]) == np.argmax(y[i]):
            correct += 1
    print('\nTest Accuracy: ', correct/len(predictions))
    print('Quadratic Weighted Kappa: ', cohen_kappa_score(y_true, y_pred, weights='quadratic'))


def pred_list(list):
    '''Creates list of predictions from one-hot predictions by the model.

    Usage: pred_list(predictions)
    '''

    result = []
    for i in range(len(list)):
        result.append(np.argmax(list[i]))
    return result


def predict_majority(model, x, y):
    '''Augments all samples of the original data, and chooses majority predictions predicted by the model.

    Usage: predict_majority(model, x_original, y_original)
    '''

    # Predict majority
    x_flip = augment(x.copy(), RandomFlip())
    x_ed = augment(x.copy(), RandomElasticDeformation())
    x_affine = augment(x.copy(), RandomAffine())
    x_noise = augment(x.copy(), RandomNoise())
    x_blur = augment(x.copy(), RandomBlur())

    y_true = pred_list(y)
    y_pred = pred_list(model.predict(x.copy()))
    y_flip = pred_list(model.predict(x_flip.copy()))
    y_ed = pred_list(model.predict(x_ed.copy()))
    y_affine = pred_list(model.predict(x_affine.copy()))
    y_noise = pred_list(model.predict(x_noise.copy()))
    y_blur = pred_list(model.predict(x_blur.copy()))

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


def main():
    '''Run training routine. Trains model specified in the config class.
    '''
    
    x = np.load(config.x_path)
    y = np.load(config.y_path)
    model = load_model(config.train_model)

    x_train, y_train, x_val, y_val, x_test, y_test = preprocess(x.copy(), y)

    if config.train_augment == 'random':
        x_train = random_augment(x_train.copy())
    elif config.train_augment == 'alpha':
        policy = AlphaPolicy()
        x_train = augment_imgs(x_train.copy(), policy)

    if config.train:
        model.fit(x_train, y_train,
            epochs=config.train_epochs,
            validation_split=0.1,
            validation_data=(x_val, y_val),
            batch_size=1,
            shuffle=True)
        model.save(config.train_save)

    if config.test:
        # Lowest Recorded Quadratic Weighted Kappa:  0.8694535743631882
        predict(model, x_test, y_test)


if __name__ == '__main__':
    main()