import sys, pandas
import numpy as np

from data_proc import find_data_files

if __name__ == '__main__':

    # dirs and files

    pkt_dir = sys.argv[1]
    pkt_files = find_data_files(pkt_dir, postfix='.csv')

    # sort lines in each file by the first column

    for pkt_file in pkt_files:
        p = pandas.read_csv(pkt_file, delimiter=',', skiprows=0, na_filter=False)
        v = p.values
        idx = np.argsort(v[:, 0])
        #open(pkt_file, 'w').close()
        with open(pkt_file + '_sorted', 'a') as f:
            for i in idx:
                f.write(','.join([str(item) for item in v[i, :]]) + '\n')



