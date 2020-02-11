import os, pandas, pickle, sys
import os.path as osp
import numpy as np

from sys import getsizeof
from sklearn.preprocessing import OneHotEncoder as ohe

def find_data_files(dir):
    data_files = []
    for f in os.listdir(dir):
        fp = osp.join(dir, f)
        if osp.isfile(fp) and fp.endswith('.csv'):
            data_files.append(fp)
    return data_files

def extract_values(data_file):
    p = pandas.read_csv(data_file, delimiter=',', skiprows=1)
    v = p.values
    if '20-02-2018' in data_file:
        v = v[:, 4:]
    idx = np.where(v[:, 1] == 'Protocol')[0]
    while len(idx) > 0:
        v = np.delete(v, idx[0], 0)
        print('Row {0} has been deleted'.format(idx[0]))
        idx = np.where(v[:, 1] == 'Protocol')[0]
    values = np.hstack([v[:, 1:2].astype(float), v[:, 3:-1].astype(float)])
    print(values.shape, getsizeof(values))
    labels = p.values[:, -1]
    return values, labels

def one_hot_encode(values, categories):
    value_categories = np.unique(values)
    oh = np.zeros((values.shape[0], len(categories)))
    for vc in value_categories:
        c_idx = categories.index(vc)
        idx = np.where(values == vc)[0]
        oh[idx, c_idx] = 1
    return oh

if __name__ == '__main__':

    # task

    task = sys.argv[1]

    # find data files

    data_dir = 'data/cicids2018'
    data_files = find_data_files(data_dir)
    stats_file = 'stats.pkl'
    dataset_file = 'data.pkl'

    # lists for categorical features and labels

    uprotos = []
    ulabels = []

    # min, max, mean and std

    X_min = None
    X_max = None
    X_mean = None
    X_std = None
    N = 0

    # collect stats

    if task == 'stats':
        stats = []
        pp = []
        labels = []
        for data_file in data_files[0:]:
            print(data_file)
            values, labels = extract_values(data_file)
            for label in np.unique(labels):
                if label not in ulabels:
                    ulabels.append(label)
            for proto in np.unique(values[:, 0]):
                if proto not in uprotos:
                    uprotos.append(proto)
            print(ulabels, uprotos)
            x_min = np.min(values[:, 1:], axis=0)
            x_max = np.max(values[:, 1:], axis=0)
            x_mean = np.mean(values[:, 1:], axis=0)
            x_std = np.std(values[:, 1:], axis=0)
            n = values.shape[0]
            if X_min is None:
                X_min = x_min
            else:
                X_min = np.min(np.vstack([x_min, X_min]), axis=0)
            if X_max is None:
                X_max = x_max
            else:
                X_max = np.max(np.vstack([x_max, X_max]), axis=0)
            if X_mean is None and X_std is None:
                X_mean = x_mean
                X_std = x_std
                N = n
            else:
                mu = (N * X_mean + n * x_mean) / (N + n)
                D = X_mean - mu
                d = x_mean - mu
                X_std = np.sqrt(N * (D**2 + X_std**2) + n * (d**2 + x_std**2)) / (N + n)
                N = N + n
                X_mean = mu
            print(N, X_min[0:4], X_max[0:4], X_mean[0:4], X_std[0:4])
            with open(osp.join(data_dir, stats_file), 'wb') as f:
                pickle.dump(ulabels, f)
                pickle.dump(uprotos, f)
                pickle.dump((N, X_min, X_max, X_mean, X_std), f)

    elif task == 'dataset':

        # load stats

        with open(osp.join(data_dir, stats_file), 'rb') as f:
            labels = pickle.load(f)
            protos = pickle.load(f)
            N, X_min, X_max, X_mean, X_std = pickle.load(f)

        # extract data

        x = []
        y = []
        for data_file in data_files[0:]:
            print(data_file)
            v, l = extract_values(data_file)
            x.append(np.hstack([
                one_hot_encode(v[:, 0], protos),
                (v[:, 1:] - np.ones((len(v), 1)).dot(X_mean.reshape(1,-1))) / (1e-10 + np.ones((len(v), 1)).dot(X_std.reshape(1,-1)))
            ]))
            y.append(one_hot_encode(l, labels))

        # save dataset

        x = np.vstack(x)
        y = np.vstack(y)
        print(x.shape, y.shape)
        with open(osp.join(data_dir, dataset_file), 'wb') as f:
            pickle.dump((x, y), f)

