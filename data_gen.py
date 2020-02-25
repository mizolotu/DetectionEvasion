import sys, os, pandas
import os.path as osp
import numpy as np

from data_proc import find_data_files
from time import time
from _collections import deque

def calculate_features(flow_ids, pkt_lists, packet_directions, bulk_thr=1.0, idle_thr=5.0):

    # 0 - 1519421406.460569, (timestamp)
    # 1 - '172.31.66.28',    (src ip)
    # 2 - 55780,             (src port)
    # 3 - '52.219.84.178',   (dst ip)
    # 4 - 443,               (dst port)
    # 5 - 6,                 (protocol)
    # 6 - 5782,              (total size)
    # 7- 40,                 (header size)
    # 8 - 16,                (flags)
    # 9 - 256                (window size)

    features = []
    for flow_id, pkt_list, pkt_dirs in zip(flow_ids, pkt_lists, packet_directions):
        pkts = np.array(pkt_list, ndmin=2)
        flags = ''.join([str(item) for sublist in [decode_tcp_flags_value(pkt[8]) for pkt in pkt_list] for item in sublist])
        fw_pkts = np.array([pkt for pkt,d in zip(pkt_list,pkt_dirs) if d > 0])
        fw_flags = ''.join([str(item) for sublist in [decode_tcp_flags_value(pkt[8]) for pkt, d in zip(pkt_list, pkt_dirs) if d > 0] for item in sublist])
        if len(fw_pkts) > 1:
            fwt = np.zeros(len(fw_pkts))
            fwt[1:] = fw_pkts[1:, 0] - fw_pkts[:-1, 0]
            fw_blk_idx = np.where(fwt <= bulk_thr)[0]
            fw_bulk = fw_pkts[fw_blk_idx, :]
            fw_blk_dur = np.sum(fwt[fw_blk_idx])
        elif len(fw_pkts) == 1:
            fw_bulk = [fw_pkts[0, :]]
            fw_blk_dur = 0
        else:
            fw_bulk = []
            fw_blk_dur = 0
        fw_bulk = np.array(fw_bulk)
        bw_pkts = np.array([pkt for pkt, d in zip(pkt_list, pkt_dirs) if d < 0])
        bw_flags = ''.join([str(item) for sublist in [decode_tcp_flags_value(pkt[8]) for pkt, d in zip(pkt_list, pkt_dirs) if d < 0] for item in sublist])
        if len(bw_pkts) > 1:
            bwt = np.zeros(len(bw_pkts))
            bwt[1:] = bw_pkts[1:, 0] - bw_pkts[:-1, 0]
            bw_blk_idx = np.where(bwt <= bulk_thr)[0]
            bw_bulk = bw_pkts[bw_blk_idx, :]
            bw_blk_dur = np.sum(bwt[bw_blk_idx])
        elif len(bw_pkts) == 1:
            bw_bulk = [bw_pkts[0, :]]
            bw_blk_dur = 0
        else:
            bw_bulk = []
            bw_blk_dur = 0
        bw_bulk = np.array(bw_bulk)

        # calculate features

        fl_dur = pkts[-1, 0] - pkts[0, 0]
        tot_fw_pk = len(fw_pkts)
        tot_bw_pk = len(bw_pkts)
        tot_l_fw_pkt = np.sum(fw_pkts[:, 6]) if len(fw_pkts) > 0 else 0

        fw_pkt_l_max = np.max(fw_pkts[:, 6]) if len(fw_pkts) > 0 else 0
        fw_pkt_l_min = np.min(fw_pkts[:, 6]) if len(fw_pkts) > 0 else 0
        fw_pkt_l_avg = np.mean(fw_pkts[:, 6]) if len(fw_pkts) > 0 else 0
        fw_pkt_l_std = np.std(fw_pkts[:, 6]) if len(fw_pkts) > 0 else 0

        bw_pkt_l_max = np.max(bw_pkts[:, 6]) if len(bw_pkts) > 0 else 0
        bw_pkt_l_min = np.min(bw_pkts[:, 6]) if len(bw_pkts) > 0 else 0
        bw_pkt_l_avg = np.mean(bw_pkts[:, 6]) if len(bw_pkts) > 0 else 0
        bw_pkt_l_std = np.std(bw_pkts[:, 6]) if len(bw_pkts) > 0 else 0

        fl_byt_s = np.sum(pkts[:, 6]) / fl_dur if fl_dur > 0 else -1
        fl_pkt_s = len(pkts) / fl_dur if fl_dur > 0 else -1

        fl_iat_avg = np.mean(pkts[1:, 0] - pkts[:-1, 0]) if len(pkts) > 1 else 0
        fl_iat_std = np.std(pkts[1:, 0] - pkts[:-1, 0]) if len(pkts) > 1 else 0
        fl_iat_max = np.max(pkts[1:, 0] - pkts[:-1, 0]) if len(pkts) > 1 else 0
        fl_iat_min = np.min(pkts[1:, 0] - pkts[:-1, 0]) if len(pkts) > 1 else 0

        fw_iat_tot = np.sum(fw_pkts[1:, 0] - fw_pkts[:-1, 0]) if len(fw_pkts) > 1 else 0
        fw_iat_avg = np.mean(fw_pkts[1:, 0] - fw_pkts[:-1, 0]) if len(fw_pkts) > 1 else 0
        fw_iat_std = np.std(fw_pkts[1:, 0] - fw_pkts[:-1, 0]) if len(fw_pkts) > 1 else 0
        fw_iat_max = np.max(fw_pkts[1:, 0] - fw_pkts[:-1, 0]) if len(fw_pkts) > 1 else 0
        fw_iat_min = np.min(fw_pkts[1:, 0] - fw_pkts[:-1, 0]) if len(fw_pkts) > 1 else 0

        bw_iat_tot = np.sum(bw_pkts[1:, 0] - bw_pkts[:-1, 0]) if len(bw_pkts) > 1 else 0
        bw_iat_avg = np.mean(bw_pkts[1:, 0] - bw_pkts[:-1, 0]) if len(bw_pkts) > 1 else 0
        bw_iat_std = np.std(bw_pkts[1:, 0] - bw_pkts[:-1, 0]) if len(bw_pkts) > 1 else 0
        bw_iat_max = np.max(bw_pkts[1:, 0] - bw_pkts[:-1, 0]) if len(bw_pkts) > 1 else 0
        bw_iat_min = np.min(bw_pkts[1:, 0] - bw_pkts[:-1, 0]) if len(bw_pkts) > 1 else 0

        fw_psh_flag = fw_flags.count('3') if len(fw_flags) > 0 else 0
        bw_psh_flag = bw_flags.count('3') if len(fw_flags) > 0 else 0
        fw_urg_flag = fw_flags.count('5') if len(bw_flags) > 0 else 0
        bw_urg_flag = bw_flags.count('5') if len(bw_flags) > 0 else 0

        fw_hdr_len = np.sum(fw_pkts[:, 7]) if len(fw_pkts) > 0 else 0
        bw_hdr_len = np.sum(bw_pkts[:, 7]) if len(bw_pkts) > 0 else 0

        if len(fw_pkts) > 0:
            fw_dur = fw_pkts[-1, 0] - fw_pkts[0, 0]
            fw_pkt_s = len(fw_pkts) / fw_dur if fw_dur > 0 else -1
        else:
            fw_pkt_s = 0
        if len(bw_pkts) > 0:
            bw_dur = bw_pkts[-1, 0] - bw_pkts[0, 0]
            bw_pkt_s = len(bw_pkts) / bw_dur if bw_dur > 0 else -1
        else:
            bw_pkt_s = 0

        pkt_len_min = np.min(pkts[:, 6])
        pkt_len_max = np.max(pkts[:, 6])
        pkt_len_avg = np.mean(pkts[:, 6])
        pkt_len_std = np.std(pkts[:, 6])

        fin_cnt = flags.count('0')
        syn_cnt = flags.count('1')
        rst_cnt = flags.count('2')
        psh_cnt = flags.count('3')
        ack_cnt = flags.count('4')
        urg_cnt = flags.count('5')
        cwe_cnt = flags.count('6')
        ece_cnt = flags.count('7')

        down_up_ratio = len(bw_pkts) / len(fw_pkts) if len(fw_pkts) > 0 else -1

        fw_byt_blk_avg = np.mean(fw_bulk[:, 6]) if len(fw_bulk) > 0 else 0
        fw_pkt_blk_avg = len(fw_bulk)
        fw_blk_rate_avg = np.sum(fw_bulk[:, 6]) / fw_blk_dur if fw_blk_dur > 0 else -1
        bw_byt_blk_avg = np.mean(bw_bulk[:, 6]) if len(bw_bulk) > 0 else 0
        bw_pkt_blk_avg = len(bw_bulk)
        bw_blk_rate_avg = np.sum(bw_bulk[:, 6]) / bw_blk_dur if bw_blk_dur > 0 else -1

        subfl_fw_pk = len(fw_pkts) / (len(fw_pkts) - len(fw_bulk)) if len(fw_pkts) - len(fw_bulk) > 0 else -1
        subfl_fw_byt = np.sum(fw_pkts[:, 6]) / (len(fw_pkts) - len(fw_bulk)) if len(fw_pkts) - len(fw_bulk) > 0 else -1
        subfl_bw_pk = len(bw_pkts) / (len(bw_pkts) - len(bw_bulk)) if len(bw_pkts) - len(bw_bulk) > 0 else -1
        subfl_bw_byt = np.sum(bw_pkts[:, 6]) / (len(bw_pkts) - len(bw_bulk)) if len(bw_pkts) - len(bw_bulk) > 0 else -1

        fw_win_byt = fw_pkts[0, 9] if len(fw_pkts) > 0 else 0
        bw_win_byt = bw_pkts[0, 9] if len(bw_pkts) > 0 else 0

        fw_act_pkt = len([pkt for pkt in fw_pkts if pkt[5] == 6 and pkt[6] > pkt[7] + 14])
        fw_seg_min = np.min(fw_pkts[:, 7]) if len(fw_pkts) > 0 else 0

        # append to the feature list

        features.append([
            fl_dur,
            tot_fw_pk,
            tot_bw_pk,
            tot_l_fw_pkt,
            fw_pkt_l_max,
            fw_pkt_l_min,
            fw_pkt_l_avg,
            fw_pkt_l_std,
            bw_pkt_l_max,
            bw_pkt_l_min,
            bw_pkt_l_avg,
            bw_pkt_l_std,
            fl_byt_s,
            fl_pkt_s,
            fl_iat_avg,
            fl_iat_std,
            fl_iat_max,
            fl_iat_min,
            fw_iat_tot,
            fw_iat_avg,
            fw_iat_std,
            fw_iat_max,
            fw_iat_min,
            bw_iat_tot,
            bw_iat_avg,
            bw_iat_std,
            bw_iat_max,
            bw_iat_min,
            fw_psh_flag,
            bw_psh_flag,
            fw_urg_flag,
            bw_urg_flag,
            fw_hdr_len,
            bw_hdr_len,
            fw_pkt_s,
            bw_pkt_s,
            pkt_len_min,
            pkt_len_max,
            pkt_len_avg,
            pkt_len_std,
            fin_cnt,
            syn_cnt,
            rst_cnt,
            psh_cnt,
            ack_cnt,
            urg_cnt,
            cwe_cnt,
            ece_cnt,
            down_up_ratio,
            fw_byt_blk_avg,
            fw_pkt_blk_avg,
            fw_blk_rate_avg,
            bw_byt_blk_avg,
            bw_pkt_blk_avg,
            bw_blk_rate_avg,
            subfl_fw_pk,
            subfl_fw_byt,
            subfl_bw_pk,
            subfl_bw_byt,
            fw_win_byt,
            bw_win_byt,
            fw_act_pkt,
            fw_seg_min,

        ])

    return features

def clean_flow_buffer(flow_ids, flow_pkts, flow_dirs):
    flow_ids_new = []
    flow_pkts_new = []
    flow_dirs_new = []
    for fi, fp, fd in zip(flow_ids, flow_pkts, flow_dirs):


def extract_flows(pkts, step=1, window=5):
    flows = []
    pkts = pkts[np.argsort(pkts[:, 0]), :]
    flow_features = []
    tracked_flow_ids = []
    tracked_flow_packets = []
    tracked_flow_directions = []
    time_min = np.floor(np.min(pkts[:, 0]))
    time_max = np.ceil(np.max(pkts[:, 0]))
    id_idx = np.array([1, 2, 3, 4, 5])
    reverse_id_idx = np.array([3, 4, 1, 2, 5])
    window_flow_ids = deque(maxlen=window)
    step_flow_ids = []
    t = time_min + 1
    for i, pkt in enumerate(pkts):
        if pkt[0] > t or i == len(pkts) - 1:
            window_flow_ids.append(step_flow_ids)
            #ids = []
            #for step_ids in window_flow_ids:
            #    for id in step_ids:
            #        spl = id.split('-')
            #        reverse_id = '-'.join(spl[2:4] + spl[0:2] + spl[4:5])
            #        if id not in ids and reverse_id not in ids:
            #            ids.append(id)
            #tracked_flow_ids_new = []
            #tracked_flow_packets_new = []
            #tracked_flow_directions_new = []
            #for id,lp,pd in zip(tracked_flow_ids, tracked_flow_packets, tracked_flow_directions):
            #    spl = id.split('-')
            #    reverse_id = '-'.join(spl[2:4] + spl[0:2] + spl[4:5])
            #    if id in ids or reverse_id in ids:
            #        tracked_flow_ids_new.append(id)
            #        tracked_flow_packets_new.append(lp)
            #        tracked_flow_directions_new.append(pd)
            tracked_flow_ids = tracked_flow_ids_new.copy()
            tracked_flow_packets = tracked_flow_packets_new.copy()
            tracked_flow_directions = tracked_flow_directions_new.copy()
            features = calculate_features(tracked_flow_ids, tracked_flow_packets, tracked_flow_directions)
            flows.extend(features)
            step_flow_ids = []
            t = int(pkt[0]) + step
        id = '-'.join([str(item) for item in pkt[id_idx]])
        reverse_id = '-'.join([str(item) for item in pkt[reverse_id_idx]])
        if id not in step_flow_ids and reverse_id not in step_flow_ids:
            step_flow_ids.append(id)
        if id not in tracked_flow_ids and reverse_id not in tracked_flow_ids:
            tracked_flow_ids.append(id)
            tracked_flow_packets.append([pkt])
            tracked_flow_directions.append([1])
        else:
            if id in tracked_flow_ids:
                direction = 1
                idx = tracked_flow_ids.index(id)
            else:
                direction = -1
                idx = tracked_flow_ids.index(reverse_id)
            tracked_flow_packets[idx].append(pkt)
            tracked_flow_directions[idx].append(direction)
    return flow_features

if __name__ == '__main__':

    # dirs and files

    pkt_dir = sys.argv[1]
    flow_dir = osp.join('/'.join(pkt_dir.split('/')[:-1]), 'flows')
    if not osp.isdir(flow_dir): os.mkdir(flow_dir)
    pkt_files = find_data_files(pkt_dir, postfix='.csv')

    # extract flow features

    for pkt_file in pkt_files:
        t_start = time()
        p = pandas.read_csv(pkt_file, delimiter=',', skiprows=0)
        v = p.values
        flows = extract_flows(v)
        print(pkt_file, v.shape, time() - t_start)
