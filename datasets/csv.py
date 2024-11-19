import pandas as pd
import numpy as np
import lightning as L

class CSV():

    class Data:
        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, filepath):
        trn, val, tst = load_data_normalised(filepath)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]
        self.color = "#ff00ff"
        self.image_size = None


def load_data_normalised(filepath):

    # load and partition data
    data = pd.read_csv(filepath).values
    data_per_year = len(data) // 10
    train_data = data[:8*data_per_year]                     
    val_data = data[8*data_per_year:9*data_per_year]        
    test_data = data[9*data_per_year:]                      
    
    # normalize data with mean and std from train + val
    mu, s = np.mean(data[:9*data_per_year], axis=0), np.std(data[:9*data_per_year], axis=0)
    train_data = (train_data - mu) / s
    val_data = (val_data - mu) / s
    test_data = (test_data - mu) / s

    # shuffle data
    rng = np.random.default_rng(seed=1)
    rng.shuffle(train_data, axis=0)
    rng.shuffle(val_data, axis=0)
    rng.shuffle(test_data, axis=0)

    return train_data, val_data, test_data