import sys
import numpy as np


from env import DeEnv
from time import sleep

def create_env(port, remote, attack, state_height):
    return lambda : DeEnv(port, remote, attack, state_height)

if __name__ == '__main__':
    server_ip = sys.argv[1]
    env = create_env(12345, (server_ip, 80), 'bruteforce', 64)
    myenv = env()
    print(myenv.action_space, myenv.observation_space)
    sleep(1)
    obs = myenv.reset()
    myenv.step(np.random.rand(4))