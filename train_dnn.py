import tensorflow as tf
import numpy as np

from data_proc import load_dataset

def create_model(nfeatures, nlayers, nhidden, ncategories, batch=512):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(nfeatures,)))
    for _ in range(nlayers):
        model.add(tf.keras.layers.Dense(nhidden, activation='relu'))
    model.add(tf.keras.layers.Dense(ncategories, activation='softmax'))
    model.compile(loss='categorical_crossentropy')
    return model

if __name__ == '__main__':

    # load data

    data, stats = load_dataset('data/cicids2018', 'data', '.pkl', 'stats.pkl')
    ulabels = stats[0]
    print(stats[1])
    nfeatures = data.shape[1] - len(ulabels)

    # select data

    labels = ulabels # ['Benign', 'Brute Force -Web']
    nsamples = [10000 for _ in labels]
    X, Y = [], []
    for i in range(len(labels)):
        if labels[i] in ulabels:
            lidx = ulabels.index(labels[i])
            idx = np.where(data[:, nfeatures + lidx]==1)[0]
            X.append(data[idx[:nsamples[i]], :nfeatures])
            y = np.zeros((len(idx[:nsamples[i]]), len(labels)))
            y[:, i] = 1
            Y.append(y)
    X = np.vstack(X)
    Y = np.vstack(Y)
    print(X.shape, Y.shape)

    # test models

    model_dir = 'models/dnn_{0}_{1}'
    n_layers = [2]
    n_hidden = [64]
    validation_split = 0.8
    for nl in n_layers:
        for nh in n_hidden:
            model = create_model(nfeatures, nl, nh, len(labels))
            model.fit(X, Y, validation_split=validation_split, verbose=True)

