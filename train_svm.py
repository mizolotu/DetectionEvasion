import numpy as np

from sklearn.svm import SVC
from data_proc import load_dataset
from joblib import dump, load

def ga_iteration(kernel, penalty, features, x_tr, y_tr, x_val, y_val):
    m, n = features.shape
    new_features = np.zeros((3 * m, n))
    new_features[:m, :] = features
    for i in range(m):
        p1 = features[np.random.randint(0, m), :]
        p2 = features[np.random.randint(0, m), :]
        co = np.random.randint(0, 2, (n,))
        new_features[m + i, np.where(co == 0)[0]] = p1[np.where(co == 0)[0]]
        new_features[m + i, np.where(co == 1)[0]] = p2[np.where(co == 1)[0]]
        mut = np.random.randint(0, 2, (n,))
        p = features[np.random.randint(0, m), :]
        new_features[2 * m + i, np.where(mut == 1)[0]] = 1 - p[np.where(mut == 1)[0]]
    f = np.zeros(3 * m)
    for i in range(3 * m):
        idx = np.where(new_features[i, :] == 1)[0]
        if len(idx) > 0:
            model = SVC(kernel=kernel, C=penalty, cache_size=4096)
            model.fit(x_tr[:, idx], y_tr)
            f[i] = model.score(x_val[:, idx], y_val)
    features_selected = new_features[np.argsort(f)[-m:], :]
    return features_selected, np.sort(f)[-m:]

if __name__ == '__main__':

    # load data

    X_tr, Y_tr, X_val, Y_val, X_te, Y_te = load_dataset('data/cicids2018', 'data', '.pkl', 'stats.pkl')
    nfeatures = X_tr.shape[1]
    nlabels = np.max(Y_tr) + 1
    print(X_tr.shape, Y_tr.shape, X_val.shape, Y_val.shape, X_te.shape, Y_te.shape)

    # lazy labeling: 0 or 1

    B_tr = Y_tr.copy()
    B_val = Y_val.copy()
    B_te = Y_te.copy()
    for b in [B_tr, B_val, B_te]:
        b[b > 0] = 1

    # test models

    model_checkpoint_path = 'models/svm_{0}_{1}_{2}/ckpt'
    model_stats_file = 'models/svm_{0}_{1}_{2}/metrics.txt'
    nsamples = X_tr.shape[0]
    n_ga_iterations = 1
    sample_size = 1000 # int(nsamples * 0.0001)
    population_size = 5
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    penalties = [0.01, 0.1, 1.0, 10.0, 100.0]
    n_labels = [2, nlabels]
    features = np.vstack([
        np.ones((1, nfeatures)),
        np.random.randint(0, 2, (population_size - 1, nfeatures))
    ])
    for kernel in kernels:
        for penalty in penalties:
            for nn in n_labels:
                for g in range(n_ga_iterations):
                    sample_idx = np.random.choice(nsamples, sample_size, replace=False)
                    x_tr = X_tr[sample_idx, :]
                    if nn == 2:
                        features, f = ga_iteration(kernel, penalty, features, x_tr, B_tr[sample_idx], X_val, B_val)
                    else:
                        features, f = ga_iteration(kernel, penalty, features, x_tr, Y_tr[sample_idx], X_val, Y_val)
                    print(g, np.max(f), np.sum(features[-1, :]))
                idx = np.where(features[-1, :] == 1)[0]
                model = SVC(kernel=kernel, C=penalty, cache_size=4096, verbose=1)
                if nn == 2:
                    model.fit(X_tr[:, idx], B_tr)
                    score = model.score(X_te[:, idx], B_te)
                else:
                    model.fit(X_tr[:, idx], Y_tr)
                    P_te = model.predict(X_te[:, idx])
                    P_te[P_te > 0] = 1
                    score = len(np.where(P_te == B_te)[0]) / len(P_te)
                dump(model, model_checkpoint_path.format(kernel, penalty, nn))
                with open(model_stats_file.format(kernel, penalty, nn), 'w') as f:
                    f.write(score)

