import numpy as np

from sklearn.svm import SVC
from data_proc import load_dataset

if __name__ == '__main__':

    # load data

    data, stats = load_dataset('data/cicids2018', 'data', '.pkl', 'stats.pkl')
    ulabels = stats[0]
    nfeatures = data.shape[1] - len(ulabels)

    # select data

    labels = ulabels # ['Benign', 'Brute Force -Web']
    nsamples = [-1 for _ in labels]
    X, Y = [], []
    for i in range(len(labels)):
        if labels[i] in ulabels:
            lidx = ulabels.index(labels[i])
            idx = np.where(data[:, nfeatures + lidx]==1)[0]
            X.append(data[idx[:nsamples[i]], :nfeatures])
            #y = np.zeros((len(idx[:nsamples[i]]), len(labels)))
            #y[:, i] = 1
            y = np.ones(len(idx[:nsamples[i]])) * lidx
            Y.append(y)
    X = np.vstack(X)
    Y = np.hstack(Y)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx, :]
    Y = Y[idx]
    B = Y.copy()
    B[B > 0] = 1
    print(X.shape, Y.shape, B.shape, np.sum(Y, axis=0, dtype=int))

    # test models

    ntrain = int(0.1 * X.shape[0])
    nvalidation = int(0.2 * X.shape[0])
    model = SVC(cache_size=4096)
    model.fit(X[:ntrain, :], B[:ntrain])
    acc = model.score(X[nvalidation:, :], B[nvalidation:])
    print(acc)