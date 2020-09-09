
# The six following lines aims to ignore the numerous warnings of tensorflow. They can be removed.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym

from mpi4py import MPI

from callbacks import MyCallback
from algorithms import DDPG, DDPG_HER, SAC, SAC_HER

from stable_baselines.her.utils import HERGoalEnvWrapper
from stable_baselines.common import set_global_seeds
from stable_baselines import logger


def train(reward_type, is_HER, env_name, seed=0):

    timesteps_per_episode = 50
    episode_per_epoch = 19*2*50
    nb_epoch = 20
    timesteps_per_epoch = timesteps_per_episode*episode_per_epoch # 95 000
    total_timesteps = timesteps_per_epoch*nb_epoch # 4 750 000

    rank = MPI.COMM_WORLD.Get_rank()

    workerseed = seed + 10000 * rank
    set_global_seeds(workerseed)
    # Create a leanring environnement,
    # and another identical environnement for evaluation
    env = gym.make(env_name, reward_type=reward_type)

    # Create a callback that evaluate the policy every n timesteps
    exp_name = "my_exp_name/{}/DDPG{}/{}".format(env_name, "HER" if is_HER else "", reward_type)
    eval_callback = MyCallback(env, timesteps_per_episode, episode_per_epoch, exp_name=exp_name, verbose=1 if rank==0 else 0)

    if not is_HER:
        env = HERGoalEnvWrapper(env)
        model = DDPG('MlpPolicy', env, timesteps_per_episode, episode_per_epoch)

    else:
        model = DDPG_HER('MlpPolicy', env, timesteps_per_episode, episode_per_epoch)

    env.seed(workerseed)

    # Run the learning process
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback) # log every n epoch

    env.close()
    del env

if __name__=='__main__':
    for env_name in ['FetchReach-v1', 'FetchSlide-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1']:
        for reward_type in ['sparse', 'dense']:
            for is_HER in [True, False]:
                train(reward_type, is_HER, env_name)
