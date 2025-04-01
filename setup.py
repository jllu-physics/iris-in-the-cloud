import os
import json
import argparse
import numpy as np
from datetime import datetime
import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing
import util

N_FEATURES = 4
N_CLASSES = 3
RNG_SEED = 42
VALID_RATIO = 0.1
TEST_RATIO = 0.2
HYPERPARAMETER_TABLE_NAME = 'hyperparameter_table.csv'

def download_dataset():
    iris = sklearn.datasets.load_iris()
    X = iris['data']
    y = iris['target']
    return X, y

def onehot_encode(y):
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(y.reshape(-1,1))
    y_onehot = enc.transform(y.reshape(-1,1)).toarray().astype(int)
    return y_onehot

def train_valid_test_split(data, valid_ratio = 0.1, test_ratio = 0.2):
    train_ratio = 1 - valid_ratio - test_ratio
    data_train_valid, data_test = \
        sklearn.model_selection.train_test_split(
            data, test_size = test_ratio
        )
    data_train, data_valid = \
        sklearn.model_selection.train_test_split(
            data_train_valid, 
            test_size = valid_ratio / (valid_ratio + train_ratio)
        )
    return data_train, data_valid, data_test

def save_feature_statistics(X, output_filename):
    feature_means = X.mean(axis=0)
    feature_stds = X.std(axis=0)

    # To avoid divide-by-zero
    feature_stds[np.where(feature_stds<1e-3)] = 1.0

    with open(output_filename, 'w') as f:
        json.dump({'feature_means': feature_means.tolist(), 'feature_stds': feature_stds.tolist()}, f)


def create_hyperparam_table(table_name = HYPERPARAMETER_TABLE_NAME):
    headers = "datetime,n_neurons,activation,optimizer,train_acc,train_loss,valid_acc,valid_loss,checkpoint\n"
    if util.if_file_exist(table_name):
        with open(table_name, 'r') as f:
            clean_first_line = f.readline().strip()
            if clean_first_line == headers:
                return
    with open(table_name, 'w') as f:
        f.writelines([headers,])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Please specify the options for setup:")
    parser.add_argument(
        "--seed", 
        type = int, 
        help = "random number seed for data split",
        default = RNG_SEED
    )
    parser.add_argument(
        "--model_version",
        type = str,
        help = 'model version',
        default = datetime.now().strftime('%Y_%m_%d')
    )
    parser.add_argument(
        "--valid_ratio",
        type = int,
        help = 'validation set ratio',
        default = VALID_RATIO
    )
    parser.add_argument(
        "--test_ratio",
        type = int,
        help = 'test set ratio',
        default = TEST_RATIO
    )
    args = parser.parse_args()
    seed = args.seed
    version = args.model_version
    valid_ratio = args.valid_ratio
    test_ratio = args.test_ratio
    data_path = os.path.join('.','model_' + version + '_assets','data')
    os.makedirs(data_path, exist_ok=True)
    X, y = download_dataset()
    y = onehot_encode(y)
    data = np.hstack((X,y))
    data_train, data_valid, data_test = \
        train_valid_test_split(data, valid_ratio=valid_ratio, test_ratio=test_ratio)
    save_feature_statistics(data_train[:,:N_FEATURES], os.path.join('.','model_' + version + '_assets',"feature_statistics.json"))
    np.savetxt(os.path.join(data_path,'data_train.csv'), data_train, delimiter = ',')
    np.savetxt(os.path.join(data_path,'data_valid.csv'), data_valid, delimiter = ',')
    np.savetxt(os.path.join(data_path,'data_test.csv'), data_test, delimiter = ',')
    create_hyperparam_table(os.path.join('.','model_' + version + '_assets',HYPERPARAMETER_TABLE_NAME))
    
    config = {
        'date and time': datetime.now().strftime('%m/%d/%Y %H:%M:%S'),
        'data split rng seed': seed,
        'valid ratio': valid_ratio,
        'test ratio': test_ratio
    }
    with open(os.path.join('.','model_' + version + '_assets','config.json'), 'w') as f:
        json.dump(config, f)