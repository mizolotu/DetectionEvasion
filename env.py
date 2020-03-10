import gym, socket, pickle, os
import numpy as np
import os.path as osp

from gym import spaces
from threading import Thread
from pcap_proc import read_iface,  calculate_features
from time import time, sleep
from train_dnn import create_model as dnn_model

class DeEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, src_port, server, attack, obs_len, iface='enp0s25', flow_std_file='data/flows/stats.pkl', model_dir='models', model_type='dnn'):

        super(DeEnv, self).__init__()

        self.obs_len = obs_len
        self.n_obs_features = 12
        self.n_actions = 4
        self.port = src_port
        self.remote = server
        self.pkt_list = []
        self.monitor_thr = Thread(target=read_iface, args=(self.pkt_list, iface, src_port, server[0]), daemon=True)
        self.monitor_thr.start()
        self.t_start = time()
        with open(flow_std_file, 'rb') as f:
            stats = pickle.load(f)
        self.target_features = np.where(stats[4] > 0)[0]
        self.xmin = stats[1]
        self.xmax = stats[2]
        self.target_model = self._load_model(model_dir, model_type)

        # actions: break, delay, pad, packet

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_actions,))

        # observation features: time, frame length, header size, window, 8 tcp flags

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.obs_len, self.n_obs_features))

    def step(self, action):
        y = self._classify()
        print(y)

    def reset(self):
        self.pkt_list.clear()
        self.t_start = time()
        sckt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sckt.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sckt.bind(('', self.port))
        sckt.connect(self.remote)
        while len(self.pkt_list) < 3:
            sleep(0.001)
        obs = self._get_obs()
        return obs

    def render(self, mode='human', close=False):
        pass

    def _get_obs(self):
        obs = np.zeros((self.obs_len, self.n_obs_features))
        for i,p in enumerate(self.pkt_list[::-1]):
            obs[-i-1, 0] = p[0] - self.t_start
            obs[-i-1, 1] = p[6]
            obs[-i-1, 2] = p[7]
            obs[-i-1, 3] = p[9]
            obs[-i-1, 4] = str(p[8]).count('0')
            obs[-i-1, 5] = str(p[8]).count('1')
            obs[-i-1, 6] = str(p[8]).count('2')
            obs[-i-1, 7] = str(p[8]).count('3')
            obs[-i-1, 8] = str(p[8]).count('4')
            obs[-i-1, 9] = str(p[8]).count('5')
            obs[-i-1, 10] = str(p[8]).count('6')
            obs[-i-1, 11] = str(p[8]).count('7')
            if i >= self.obs_len - 1:
                break
        return obs

    def _load_model(self, model_dir, prefix):
        model_score = 0
        model_name = ''
        for sd in os.listdir(model_dir):
            subdir = osp.join(model_dir, sd)
            if osp.isdir(subdir) and sd.startswith(prefix):
                try:
                    with open(osp.join(subdir, 'metrics.txt'), 'r') as f:
                        line = f.readline()
                        spl = line.split(',')
                        score = float(spl[-1])
                        if score > model_score:
                            model_score = score
                            model_name = sd
                            print(model_name, model_score)
                except Exception as e:
                    print(e)
        if model_name.startswith('dnn'):
            params = [int(item) for item in model_name.split('_')[1:]]
            model = dnn_model(len(self.target_features), *params)
        return model

    def _classify(self):
        flow_ids = ['-{0}-{1}-{2}-6'.format(self.port, self.remote[0], self.remote[1])]
        pkt_lists = [[[pkt[0], pkt[6], pkt[7], pkt[9]] for pkt in self.pkt_list]]
        pkt_flags = [[str(pkt[8]) for pkt in self.pkt_list]]
        pkt_directions = [[1 if pkt[2] == self.port else -1 for pkt in self.pkt_list]]
        v = calculate_features(flow_ids, pkt_lists, pkt_flags, pkt_directions)
        x = (np.array(v[0, self.target_features]) - self.xmin[self.target_features]) / (self.xmax[self.target_features] - self.xmin[self.target_features])
        return self.target_model.predict(x)

