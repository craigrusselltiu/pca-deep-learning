class Config():

    def __init__(self):
        self.x_path = "x_train.npy"
        self.y_path = "y_train.npy"

        self.aa_search_epochs = 36
        self.aa_train_epochs = 25
        self.aa_val_split = 0.1
        self.aa_transforms = ["flip", "affine", "noise", "blur", "elasticD"]
        self.aa_seed = 69420
        self.aa_model = "base_64"
        self.aa_batch = 1
        self.aa_log = str(self.aa_seed) + "_best_policies.txt"