import sys, os, pandas
import os.path as osp
import numpy as np

from data_proc import find_data_files

def extract_flows(pkts, step=1, window=5):
    flow_features = []
    tracked_flow_ids = []
    tracked_flow_packets = []
    time_min = np.floor(np.min(pkts[:, 0]))
    time_max = np.ceil(np.max(pkts[:, 0]))
    id_idx = np.arange(1, 6)
    for t in np.arange(time_min, time_max, step):
        current_flow_ids = []
        current_flow_packets = []
        pkts_t = pkts[np.where((pkts[:, 0] >= t - window) & (pkts[:, 0] < t))[0], :]
        for i, pkt in enumerate(pkts_t):
            id = '-'.join([str(item) for item in pkt[id_idx]])
            if id not in current_flow_ids:
                current_flow_ids.append(id)
                current_flow_packets.append([pkt])
            else:
                idx = current_flow_ids.index(id)
                current_flow_packets[idx].append(pkt)
        #print(len(current_flow_ids))
    return flow_features

if __name__ == '__main__':

    # dirs and files

    pkt_dir = sys.argv[1]
    flow_dir = osp.join('/'.join(pkt_dir.split('/')[:-1]), 'flows')
    if not osp.isdir(flow_dir): os.mkdir(flow_dir)
    pkt_files = find_data_files(pkt_dir, postfix='.csv')

    # extract flow features

    for pkt_file in pkt_files:
        print(pkt_file)
        p = pandas.read_csv(pkt_file, delimiter=',', skiprows=0)
        v = p.values
        print(v.shape)
        flows = extract_flows(v)
