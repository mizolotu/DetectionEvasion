import numpy as np
import tensorflow as tf
from baselines.common.runners import AbstractEnvRunner
from threading import Thread
from queue import Queue

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run_env(self, env_idx, obs, done, q):
        scores = []
        steps = 0
        obss, actions, values, states, neglogpacs, rewards, dones = [], [], [], [], [], [], []
        for _ in range(self.nsteps):
            obs = tf.constant(obs)
            action, value, state, neglogpac = self.model.step(obs)
            actions.append(action)
            values.append(value)
            states.append(state)
            neglogpacs.append(neglogpac)
            obss.append(obs.copy())
            dones.append(done)

            obs, reward, done, info = self.env.step_env(env_idx, action)
            if 'r' in info.keys():
                scores.append(info['r'])
            if 'l' in info.keys() and info['l'] > steps:
                steps = info['l']
            rewards.append(reward)
        epinfos = {'r': np.mean(scores), 'l': steps}
        q.put((obss, actions, values, states, neglogpacs, rewards, dones, epinfos))

    def run(self):

        # Here, we init the lists that will contain the mb of experiences

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []

        thrs = []
        q = Queue()
        for env_i in range(self.env.num_envs):
            thrs.append(Thread(target=self.run_env, args=(env_i, self.obs[env_i], self.dones[env_i], q), daemon=True))
            thrs[env_i].start()
        for env_i in range(self.env.num_envs):
            thrs[env_i].join()

        for i in range(self.env.num_envs):
            obs, actions, values, states, neglogpacs, rewards, dones, epinfos = q.get()
            mb_obs.append(obs)
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_rewards.append(rewards)
            mb_dones.append(dones)
            epinfos.append(epinfos)

        #batch of steps to batch of rollouts

        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        print(mb_obs.shape, mb_rewards.shape, mb_actions.shape, mb_values.shape, mb_neglogpacs.shape, mb_dones.shape)

        last_values = self.model.value(tf.constant(self.obs))._numpy()

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
