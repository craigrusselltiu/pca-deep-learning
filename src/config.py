class Config():
    '''Configuration class for efficient setting changes.
    '''

    def __init__(self):
        self.x_path = 'data/x_train.npy'
        self.y_path = 'data/y_train.npy'
        self.img_path = '../../prostate_images/PROSTATEx'
        self.roi_x = 40
        self.roi_y = 40
        self.roi_z = 4

        self.preprocess = 'none'

        self.train = False
        self.train_augment = 'none'
        self.train_model = 'models/new_alpha_base_64' # Load model
        self.train_save = 'models/new_alpha_base_64' # Save model
        self.train_epochs = 50

        self.test = True
        self.test_epochs = 100

        self.aa_seed = 2020 # Random seed for reproducibility
        self.aa_transforms = ['flip', 'affine', 'noise', 'blur', 'elasticD']
        self.aa_model = 'models/base_64' # Child model used for training

        self.aa_search_epochs = 240 # Number of epochs for random search
        self.aa_train_epochs = 30 # Number of epochs for training each policy
        self.aa_val_split = 0.1 # Validation split for child model
        self.aa_batch = 1 # Batch size for child model
        self.aa_min_prob = 0 # Minimum probability for policy transforms 0 (0%) - 10 (100%)

        self.aa_n_best = 24 # Keep n best policies
        self.aa_log = 'logs/' + str(self.aa_seed) + '_best_policies.txt'
        