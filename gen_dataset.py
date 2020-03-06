import sys, pickle, pandas
import os.path as osp
import numpy as np

from data_proc import find_data_files

if __name__ == '__main__':


    # dataset dirs and files

    dir = sys.argv[1]
    subdirs = sys.argv[2].split(',')
    stat_file = osp.join(dir, 'stats.pkl')
    with open(stat_file, 'rb') as f:
        stats = pickle.load(f)
    feature_inds = np.where(stats[4][:-1] > 0)[0]
    files = []
    for subdir in subdirs:
        files.extend(find_data_files(osp.join(dir, subdir)))

    # generate dataset

    x = None
    y = None
    for fi,f in enumerate(files):
        print(fi / len(files), f)
        p = pandas.read_csv(f, delimiter=',', dtype=float, header=None)
        v = p.values
        if x is not None and y is not None:
            x = np.vstack([x, v[:, feature_inds]])
            y = np.hstack([y, v[:, -1]])
        else:
            x = v[:, feature_inds]
            y = v[:, -1]
        print(x.shape, y.shape, sys.getsizeof(x))
    size = sys.getsizeof(x)
    maxsize = 4e9
    nchunks = int(size // maxsize) + 1
    nsamples = x.shape[0]
    chunk_size = nsamples // nchunks
    for i in range(nchunks):
        if i == nchunks - 1:
            idx = np.arange((i - 1) * chunk_size, nsamples - 1)
        else:
            idx = np.arange((i - 1) * chunk_size, i * chunk_size)
        print(len(idx))
        x_chunk = x[idx, :]
        y_chunk = y[idx]
        with open(osp.join(dir, 'flows{0}.pkl'.format(i)), 'wb') as f:
            pickle.dump(np.hstack([x_chunk, y_chunk.reshape(-1, 1)]), f)