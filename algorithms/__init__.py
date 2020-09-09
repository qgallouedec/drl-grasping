""" 
My algorithms
"""

import stable_baselines
from stable_baselines import HER
from stable_baselines.ddpg import NormalActionNoise
import numpy as np

class DDPG(stable_baselines.DDPG):
    def __init__(self, policy, env, timesteps_per_episode, episode_per_epoch):
        timesteps_per_epoch = timesteps_per_episode*episode_per_epoch

        n_actions = env.action_space.shape[0]
        noise_std = 0.2
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std*np.ones(n_actions))
        super().__init__(
            policy, env,
            # gamma=0.98, ###########################################
            # nb_train_steps=40, ####################################
            # nb_rollout_steps=timesteps_per_epoch, #################
            # nb_eval_steps=0, ######################################
            action_noise=action_noise, ############################
            # normalize_observations=False,
            # tau=0.95, #############################################
            # batch_size=256, #######################################
            # observation_range=(-200.0, 200.0), ####################
            # actor_lr=0.001, #######################################
            # critic_lr=0.001,#######################################
            # clip_norm=5, ##########################################
            # buffer_size=1000000,###################################
            # random_exploration=0.3, ###############################
            # policy_kwargs=dict(layers=[256, 256, 256]),############
        )


            # Hyperparam from the original paper
            # Actor and critic networks: 3 layers with 256 units each and ReLU non-linearities
            # Adam optimizer (Kingma and Ba, 2014) with lr=1e−3 for training both actor and critic
            # Buffer size : 1e6 transitions
            # Polyak-averaging coefficient:0.95
            # Action L2 norm coefficient:1.0
            # Observation clipping:[−200,200]
            # Batch size:256
            # Rollouts per MPI worker:2
            # Number of MPI workers:19
            # Cycles per epoch:50
            # Batches per cycle:40
            # Test rollouts per epoch:10
            # Probability of random actions:0.3
            # Scale of additive Gaussian noise:0.2
            # Probability of HER experience replay:0.8
            # Normalized clipping:[−5,5]


class DDPG_HER(HER):
    def __init__(self, policy, env, timesteps_per_episode, episode_per_epoch):
        super().__init__(policy, env, DDPG,
            n_sampled_goal=4,
            goal_selection_strategy='future', 
            timesteps_per_episode=timesteps_per_episode,
            episode_per_epoch=episode_per_epoch)


class SAC(stable_baselines.SAC):
    def __init__(self, policy, env):
        n_actions = env.action_space.shape[0]
        noise_std = 0.2
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std*np.ones(n_actions))
        super().__init__(policy, env, 
            verbose=1, buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95, batch_size=256,
            random_exploration=0.3,
            action_noise=action_noise,
            policy_kwargs=dict(layers=[256, 256, 256]))

class SAC_HER(HER):
    def __init__(self, policy, env):
        super().__init__(policy, env, SAC, n_sampled_goal=4,
            goal_selection_strategy='future')


if __name__=='__main__':
    from stable_baselines import DDPG
    import gym
    env = gym.make("FetchReach-v1")
    my_her = DDPG_HER('MlpPolicy', env)