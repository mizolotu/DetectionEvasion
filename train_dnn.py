import tensorflow as tf
import numpy as np

from data_proc import load_dataset

def create_model(nfeatures, nlayers, nhidden, ncategories, lr=1e-6):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(nfeatures,)))
    for _ in range(nlayers):
        model.add(tf.keras.layers.Dense(nhidden, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(ncategories))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=['accuracy'])
    return model

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
    print(X.shape, Y.shape, np.sum(Y, axis=0, dtype=int))

    # test models

    model_checkpoint_path = 'models/dnn_{0}_{1}/ckpt'
    model_stats_file = 'models/dnn_{0}_{1}/metrics.txt'
    n_layers = [1, 2, 3, 4, 5]
    n_hidden = [128, 256, 512, 768, 1024]
    validation_split = 0.2
    batch_size = 512
    epochs=1000
    for nl in n_layers:
        for nh in n_hidden:
            model = create_model(nfeatures, nl, nh, len(labels))
            model.summary()
            h = model.fit(X, Y, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=True, shuffle=True)
            model.save_weights(model_checkpoint_path.format(nl, nh))
            metrics = ','.join([str(h.history[key][0]) for key in h.history.keys()])
            with open(model_stats_file.format(nl, nh), 'w') as f:
                f.write(metrics)