import numpy as np

from autoaugment import Policy
from config import Config
from imblearn.over_sampling import RandomOverSampler
from keras.models import load_model
from keras.utils import to_categorical
from random import choice
from random import randint
from random import seed
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

# Import config file
config = Config()


def update_policies(e, policies):
    '''Write to a log file to store best policies found in the search.

    Usage: update_policies(epoch, list_of_policies)
    '''

    log = open(config.aa_log, 'w')
    log.write('Seed: ' + str(config.aa_seed))
    log.write('\nEpochs ran: ' + str(e+1))
    log.write('\n\nBest policies:\n')
    for index, policy in enumerate(policies):
        log.write(str(index+1) + '. (\'' + policy.t1_input + '\', ' + str(policy.m1_input) + ', ' + str(policy.p1) + ', \'' + policy.t2_input + '\', ' + str(policy.m2_input) + ', ' + str(policy.p2) + ') | Kappa: ' + str(policy.kappa) + '\n')
    log.close()


def augment_imgs(x, policy):
    '''Augment a list of images with a given policy.

    Usage: augment_imgs(image_list, policy)
    '''

    result = x
    for i in range(len(x)):

        # Get ROI and adjust axes for augmentation
        roi = result[i]
        roi = np.swapaxes(roi, 2, 3)
        roi = np.swapaxes(roi, 1, 2)
        roi = np.swapaxes(roi, 0, 1)

        # Perform augmentation, adjust axes back, and update x_train
        output = policy(roi)
        output = np.swapaxes(output, 0, 1)
        output = np.swapaxes(output, 1, 2)
        output = np.swapaxes(output, 2, 3)
        result[i] = output
    return result


def train(m, x_train, y_train, x_val, y_val, x_test, y_test):
    '''Train child model from policy-augmented data.

    Usage: train(model, x_train, y_train, x_val, y_val, x_test, y_test)
    '''

    model = load_model(m)
    model.fit(x_train, y_train,
        epochs = config.aa_train_epochs,
        validation_split=config.aa_val_split,
        validation_data=(x_val, y_val),
        batch_size=config.aa_batch,
        shuffle=True
    )

    # Calculate kappa and return
    pred = model.predict(x_test)
    y_true = []
    y_pred = []
    for i in range(len(y_test)):
        y_true.append(np.argmax(y_test[i]))
        y_pred.append(np.argmax(pred[i]))
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    return kappa


def main():
    seed(config.aa_seed)

    x_train = np.load('data/x_adc.npy')
    y_train = np.load('data/y_adc.npy')

    # Reshape x_train to fit oversampler
    orig_shape = np.shape(x_train)
    x_train = np.reshape(x_train, (orig_shape[0], orig_shape[1] * orig_shape[2] * orig_shape[3]))

    oversample = RandomOverSampler(sampling_strategy='not majority')

    # Reshape x_train to prepare for training
    x_train, y_train = oversample.fit_resample(x_train, y_train)
    x_train = np.reshape(x_train, (len(x_train), orig_shape[1], orig_shape[2], orig_shape[3], 1))
    
    y_train = [x - 1 for x in y_train]
    y_train = to_categorical(y_train, 5)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train)

    # Search loop
    best_policies = []
    for e in range(config.aa_search_epochs):
        print('\n\n--- Random Search Epoch', e+1, '---')

        transform1 = choice(config.aa_transforms)
        m1 = randint(1, 9)
        p1 = randint(1, 10) / 10.0

        # Create new augmented choices to avoid duplicate policies
        transforms_aug = config.aa_transforms.copy()
        transforms_aug.remove(transform1)

        transform2 = choice(transforms_aug)
        m2 = randint(1, 9)
        p2 = randint(1, 10) / 10.0

        print('Chosen policy:', transform1, 'm:', m1, 'p:', p1, '|', transform2, 'm:', m2, 'p:', p2, '\n')
        cur_policy = Policy(transform1, m1, p1, transform2, m2, p2)
        x_aug = augment_imgs(x_train.copy(), cur_policy)

        cur_policy.kappa = train(config.aa_model, x_aug, y_train, x_val, y_val, x_test, y_test)
        print('\nQuadratic Weighted Kappa:', cur_policy.kappa)

        # Update best policies
        inserted = False
        for index, policy in enumerate(best_policies):
            if cur_policy.kappa > policy.kappa:
                inserted = True
                best_policies.insert(index, cur_policy)
                if len(best_policies) > config.aa_n_best:
                    best_policies.pop()
                break

        if len(best_policies) < config.aa_n_best and not inserted:
            best_policies.append(cur_policy)

        update_policies(e, best_policies)


if __name__ == '__main__':
    main()