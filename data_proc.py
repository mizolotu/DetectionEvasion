import os, pandas, pickle, sys
import os.path as osp
import numpy as np

from sys import getsizeof

def find_data_files(dir):
    data_files = []
    for f in os.listdir(dir):
        fp = osp.join(dir, f)
        if osp.isfile(fp) and fp.endswith('.csv'):
            data_files.append(fp)
    return data_files

def extract_values(data_file, minus_ids=[67, 68]):
    p = pandas.read_csv(data_file, delimiter=',', skiprows=1)
    v = p.values
    if '20-02-2018' in data_file:
        v = v[:, 4:]

    # remove header line in the middle of the data

    idx = np.where(v[:, 1] == 'Protocol')[0]
    if len(idx) > 0:
        v = np.delete(v, idx, 0)
        print('Rows {0} have been deleted'.format(idx))

    # substitute minus ones with zeroes in columns minus_ids

    for mi in minus_ids:
        v[np.where(v[:, mi] == -1)[0], mi] = 0

    # stack values

    values = np.hstack([v[:, 1:2].astype(float), v[:, 3:-1].astype(float)])
    labels = v[:, -1]

    # remove lines with nan and inf

    finites = np.all(np.isfinite(values), axis=1)
    idx = np.where(finites == False)[0]
    if len(idx) > 0:
        print('{0} non-finite rows found'.format(len(idx)))
        values = np.delete(values, idx, 0)
        labels = np.delete(labels, idx)

    # remove lines with negative values

    negatives = np.all(values >= 0, axis=1)
    idx = np.where(negatives == False)[0]
    if len(idx) > 0:
        print('{0} non-positive rows found'.format(len(idx)))
        values = np.delete(values, idx, 0)
        labels = np.delete(labels, idx)

    print(values.shape, labels.shape, getsizeof(values))

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

    # args

    n_data_files = int(sys.argv[1])
    tasks = sys.argv[2:]

    # find data files

    data_dir = 'data/cicids2018'
    data_files = find_data_files(data_dir)
    stats_file = 'stats.pkl'
    dataset_file = 'data{0}.pkl'

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

    if 'stats' in tasks:
        stats = []
        pp = []
        labels = []
        for data_file in data_files[0:n_data_files]:
            print(data_file)
            values, labels = extract_values(data_file)
            print(np.unique(labels))
            for label in np.unique(labels):
                if label not in ulabels:
                    ulabels.append(label)
            for proto in np.unique(values[:, 0]):
                if proto not in uprotos:
                    uprotos.append(proto)
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
                X_std = np.sqrt((N * (D**2 + X_std**2) + n * (d**2 + x_std**2)) / (N + n))
                N = N + n
                X_mean = mu
            with open(osp.join(data_dir, stats_file), 'wb') as f:
                pickle.dump(ulabels, f)
                pickle.dump(uprotos, f)
                pickle.dump((N, X_min, X_max, X_mean, X_std), f)

    if 'dataset' in tasks:

        # load stats

        with open(osp.join(data_dir, stats_file), 'rb') as f:
            labels = pickle.load(f)
            protos = pickle.load(f)
            N, X_min, X_max, X_mean, X_std = pickle.load(f)

        # extract data

        x = []
        y = []
        for data_file in data_files[0:n_data_files]:
            print(data_file)
            v, l = extract_values(data_file)
            x.append(np.hstack([
                one_hot_encode(v[:, 0], protos),
                (v[:, 1 + idx] - np.ones((len(v), 1)).dot(X_mean[idx].reshape(1, -1))) / (1e-10 + np.ones((len(v), 1)).dot(X_std[idx].reshape(1, -1)))
            ]))
            y.append(one_hot_encode(l, labels))

        # save dataset

        x = np.vstack(x)
        y = np.vstack(y)
        xy = np.hstack([x, y])
        nfiles = 4
        idx = np.arange(0, xy.shape[0], xy.shape[0] // nfiles)
        for i in range(nfiles):
            fname = osp.join(data_dir, dataset_file.format(i))
            if i < nfiles - 1:
                idx_i = np.arange(idx[i], idx[i+1])
            else:
                idx_i = np.arange(idx[i], xy.shape[0])
            with open(fname, 'wb') as f:
                pickle.dump(xy[idx_i, :], f)

