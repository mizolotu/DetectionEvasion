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
    print(myenv.action_space, myenv.observation_space)
    sleep(1)
    obs = myenv.reset()
    myenv.step(np.random.rand(4))