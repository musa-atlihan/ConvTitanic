try:
   import cPickle as pickle
except:
   import pickle
import numpy as np
import pandas as pd
import theano

class dataset(object):
    """Represent a dataset.
    
    Attributes:
        dm_x (obj:`ndarray`): Represents the input x in the dataset.
        dm_y (obj:`ndarray`): Represents y in the dataset.
    """

    def __init__(self, dm_x, dm_y):
        self.x = dm_x
        self.y = dm_y

        self.n_samples = dm_x.shape[0] * 20 // 100

        self.num_test = self.n_samples

        self.num_cv = self.n_samples

        self.num_train = dm_x.shape[0] - (self.num_test + self.num_cv)

        # number of folds for the cross validation
        self.n_cv_folds = (dm_x.shape[0] - self.n_samples) // self.n_samples

        # last index number for kfold cv, update on each split.
        self.cv_fold_index = 0

    def cv_split(self, index=None):
        # Split train set into train, validation and test sets

        test_set_x = self.x[:self.n_samples,:]
        test_set_y = self.y[:self.n_samples]

        train_x = self.x[self.n_samples:,:]
        train_y = self.y[self.n_samples:]

        if index:
            cv_index = index
        else:
            cv_index = self.cv_fold_index
            if cv_index + 1 >= self.n_cv_folds:
                self.cv_fold_index = 0
            else:
                self.cv_fold_index = self.cv_fold_index + 1

        valid_set_x = train_x[cv_index * self.n_samples:(cv_index + 1) * self.n_samples,:]
        valid_set_y = train_y[cv_index * self.n_samples:(cv_index + 1) * self.n_samples]
        tsx1 = train_x[:cv_index * self.n_samples,:]
        tsx2 = train_x[(cv_index + 1) * self.n_samples:,:]
        tsy1 = train_y[:cv_index * self.n_samples]
        tsy2 = train_y[(cv_index + 1) * self.n_samples:]
        train_set_x = np.concatenate((tsx1,tsx2), axis=0)
        train_set_y = np.concatenate((tsy1,tsy2), axis=0)

        test_set_xy = (test_set_x, test_set_y)
        valid_set_xy = (valid_set_x, valid_set_y)
        train_set_xy = (train_set_x, train_set_y)

        return (test_set_xy, valid_set_xy, train_set_xy)



def get_dataset(dataset_path, name='dataset', 
                                    randomize=False, normalize=False):
    """Read the dataset from csv file and create dataset class instance.

    If randomize is True then shuffle rows. If normalize is 
    True then normalize x columns.
    """

    dm = pd.read_csv(dataset_path).as_matrix()

    if randomize:
        np.random.shuffle(dm)

    dm_x = dm[:,2:]
    dm_y = dm[:,1].astype(np.int32)

    if normalize:
        dm_x = (dm_x - dm_x.mean(axis=0)) / dm_x.std(axis=0)

    ds = dataset(dm_x, dm_y)
    
    return ds