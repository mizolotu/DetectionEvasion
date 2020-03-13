import sys
import numpy as np


from env import DeEnv
from time import sleep

def create_env(iface, port, remote, url, attack, state_height):
    return lambda : DeEnv(iface, port, remote, url, attack, state_height)

if __name__ == '__main__':
    iface = sys.argv[1]
    server_ip = sys.argv[2]
    env = create_env(iface, 12345, (server_ip, 80), '/DVWA-master/login.php', 'bruteforce', 64)
    myenv = env()
    obs = myenv.reset()
    R = 0
    for i in range(1000):
        obs, r, d, info = myenv.step(np.random.rand(3))
        R += r
        print(i, R)
        if d:
            obs = myenv.reset()
            R = 0

