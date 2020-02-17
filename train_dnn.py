import numpy as np
from data_proc import load_dataset

if __name__ == '__main__':

    # load data

    data, stats = load_dataset('data/cicids2018', 'data', '.pkl', 'stats.pkl')
    ulabels = stats[0]
    nfeatures = data.shape[1] - len(ulabels)

    # select data

    labels = ['Benign', 'Brute Force -Web']
    nsamples = [1000, 100]
    X, Y = [], []
    for i in range(len(labels)):
        idx = np.where(ulabels==labels[i])[0]
        X.append(data[idx[:nsamples[i]], :nfeatures])
        y = np.zeros((len(idx), len(labels)))
        y[:, i] = 1
        Y.append(y)
    X = np.vstack(X)
    Y = np.vstack(Y)
    print(X.shape, Y.shape)


