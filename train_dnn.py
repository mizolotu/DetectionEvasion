import tensorflow as tf
import numpy as np

from data_proc import load_dataset

class IntrusionDetectionAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='intrusion_detection_accuracy', **kwargs):
        super(IntrusionDetectionAccuracy, self).__init__(name=name, **kwargs)
        self.cc = self.add_weight(name='cc', initializer='zeros')
        self.tt = self.add_weight(name='tt', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values0 = (tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')) & (y_true == 0)
        values0 = tf.cast(values0, 'float32')
        values1 = (tf.cast(y_true, 'int32') > 0) & (tf.cast(y_pred, 'int32') > 0)
        values1 = tf.cast(values1, 'float32')
        values2 = tf.cast(y_true, 'int32') >= 0
        values2 = tf.cast(values2, 'float32')
        self.cc.assign_add(tf.reduce_sum(values0) + tf.reduce_sum(values1))
        self.tt.assign_add(tf.reduce_sum(values2))

    def result(self):
        return self.cc / self.tt

    def reset_states(self):
        self.cc.assign(0.)
        self.tt.assign(0.)


def create_model(nfeatures, nlayers, nhidden, ncategories=2, lr=1e-6):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(nfeatures,)))
    for _ in range(nlayers):
        model.add(tf.keras.layers.Dense(nhidden, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(ncategories))
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        metrics=['accuracy', IntrusionDetectionAccuracy()]
    )
    return model

if __name__ == '__main__':

    # load data

    X_tr, Y_tr, X_val, Y_val, X_te, Y_te = load_dataset('data/flows', 'flows', '.pkl')
    nfeatures = X_tr.shape[1]
    nlabels = len(np.unique(Y_tr))
    print(X_tr.dtype, Y_tr.dtype, X_tr.shape, Y_tr.shape, X_val.shape, Y_val.shape, X_te.shape, Y_te.shape)

    # lazy labeling: 0 or 1

    B_tr = Y_tr.copy()
    B_val = Y_val.copy()
    B_te = Y_te.copy()
    for b in [B_tr, B_val, B_te]:
        b[b > 0] = 1

    # test models

    model_checkpoint_path = 'models/dnn_{0}_{1}/ckpt'
    model_stats_file = 'models/dnn_{0}_{1}/metrics.txt'
    n_layers = [1, 2, 3, 4, 5]
    n_hidden = [256, 512, 768, 1024]
    n_labels = [2]
    batch_size = 512
    epochs = 1000
    for nl in n_layers:
        for nh in n_hidden:
            for nn in n_labels:
                model = create_model(nfeatures, nl, nh, nn)
                model.summary()
                if nn == 2:
                    h = model.fit(X_tr, B_tr, validation_data=(X_val, B_val), epochs=epochs, batch_size=batch_size, verbose=True, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
                    results = model.evaluate(X_te, B_te)
                else:
                    h = model.fit(X_tr, Y_tr, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, verbose=True, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
                    results = model.evaluate(X_te, Y_te)
                model.save_weights(model_checkpoint_path.format(nl, nh, nn))
                metrics = ','.join([str(h.history[key][0]) for key in h.history.keys()] + [str(r) for r in results])
                with open(model_stats_file.format(nl, nh, nn), 'w') as f:
                    f.write(metrics)