"""My callback
"""

from stable_baselines.common.callbacks import EvalCallback, BaseCallback
from stable_baselines.her.utils import HERGoalEnvWrapper

from mpi4py import MPI

from datetime import datetime
import copy

class MyCallback(EvalCallback):
    def __init__(self, env, timesteps_per_episode, episode_per_epoch, exp_name='unamed_exp', verbose=1):
        # Goal env does not work with the callback. You should use
        # a wrapper to convert the env to a standard env.
        rank = MPI.COMM_WORLD.Get_rank()
        eval_env = copy.deepcopy(env)
        eval_env = HERGoalEnvWrapper(eval_env)
        log_path='./log/{}/'.format(exp_name) + datetime.now().strftime("%m_%d_%Y__%H_%M_%f") + '/'

        timesteps_per_epoch = timesteps_per_episode*episode_per_epoch

        super().__init__(eval_env, best_model_save_path=log_path,
                        log_path=log_path, eval_freq=timesteps_per_epoch,
                        deterministic=True, render=False, n_eval_episodes=10,
                        verbose=verbose)

if __name__=='__main__':
    import gym
    env = gym.make("FetchReach-v1")
    my_callback = MyHERCallback(env)