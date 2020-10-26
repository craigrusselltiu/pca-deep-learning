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

config = Config()


def update_policies(e, policies):
    log = open(config.aa_log, "w")
    log.write("Epochs ran: " + str(e+1) + "\n\n")
    for index, policy in enumerate(policies):
        log.write(str(index+1) + ". (" + policy.t1_input + ", " + str(policy.m1_input) + ", " + str(policy.p1) + ", " + policy.t2_input + ", " + str(policy.m2_input) + ", " + str(policy.p2) + ") | Kappa: " + str(policy.kappa) + "\n")
    log.close()


def augment_imgs(x, policy):
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
    model = load_model(m)
    model.fit(x_train, y_train,
        epochs = config.aa_train_epochs,
        validation_split=config.aa_val_split,
        validation_data=(x_val, y_val),
        batch_size=config.aa_batch,
        shuffle=True
    )

    pred = model.predict(x_test)
    y_true = []
    y_pred = []
    for i in range(len(y_test)):
        y_true.append(np.argmax(y_test[i]))
        y_pred.append(np.argmax(pred[i]))
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    return kappa


def main():

    # Set random seed for reproducibility
    seed(config.aa_seed)

    # Load dataset
    x_train = np.load('x_adc.npy')
    y_train = np.load('y_adc.npy')

    # Reshape x_train to fit oversampler
    orig_shape = np.shape(x_train)
    x_train = np.reshape(x_train, (orig_shape[0], orig_shape[1] * orig_shape[2] * orig_shape[3]))

    # Oversample imbalanced classes so that all classes have equal occurrences
    oversample = RandomOverSampler(sampling_strategy='not majority')

    # Reshape x_train to prepare for training
    x_train, y_train = oversample.fit_resample(x_train, y_train)
    x_train = np.reshape(x_train, (len(x_train), orig_shape[1], orig_shape[2], orig_shape[3], 1))
    
    # One hot encoding for y_train
    y_train = [x - 1 for x in y_train]
    y_train = to_categorical(y_train, 5)

    # Split train, validate and test data
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train)

    best_policies = []

    for e in range(config.aa_search_epochs):
        print('\n\n--- Random Search Epoch', e+1, '---')

        transform1 = choice(config.aa_transforms)
        m1 = randint(0, 9)
        p1 = randint(0, 10) / 10.0

        transform2 = choice(config.aa_transforms)
        m2 = randint(0, 9)
        p2 = randint(0, 10) / 10.0

        print('Chosen policy:', transform1, "m:", m1, "p:", p1, "|", transform2, "m:", m2, "p:", p2, "\n")
        cur_policy = Policy(transform1, m1, p1, transform2, m2, p2)
        x_aug = augment_imgs(x_train.copy(), cur_policy)

        cur_policy.kappa = train(config.aa_model, x_aug, y_train, x_val, y_val, x_test, y_test)
        print('\nQuadratic Weighted Kappa:', cur_policy.kappa)

        inserted = False
        for index, policy in enumerate(best_policies):
            if cur_policy.kappa > policy.kappa:
                inserted = True
                best_policies.insert(index, cur_policy)
                if len(best_policies) > 10:
                    best_policies.pop()
                break

        if len(best_policies) < 10 and not inserted:
            best_policies.append(cur_policy)

        update_policies(e, best_policies)


if __name__ == '__main__':
    main()