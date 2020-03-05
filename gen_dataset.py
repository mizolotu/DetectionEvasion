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
    print(files)

    # generate dataset

    x = []
    y = []
    for fi,f in enumerate(files):
        p = pandas.read_csv(f, delimiter=',', dtype=float, header=None)
        v = p.values
        x.append(v[:, feature_inds])
        y.append(v[:, -1])
    x = np.vstack(x)
    y = np.hstack(y)
    print(x.shape, y.shape, sys.getsizeof(x))