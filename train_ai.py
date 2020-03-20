import sys
import numpy as np

from baselines.common.vec_env import SubprocVecEnv
from baselines.ppo2.ppo2 import learn as learn_ppo
from env import DeEnv
from time import sleep

def create_env(iface, port, remote, url, attack, state_height):
    return lambda : DeEnv(iface, port, remote, url, attack, state_height)

if __name__ == '__main__':

    # args

    iface = sys.argv[1]
    server_ip = sys.argv[2]

    # envs

    nenvs = 16
    ports = [12340 + i for i in range(nenvs)]
    env_fns = [create_env(iface, port, (server_ip, 80), '/DVWA-master/login.php', 'bruteforce', 64) for port in ports]
    env = SubprocVecEnv(env_fns)
    learn_ppo(env=env, network='mlp', nsteps=128, total_timesteps=1000000)
