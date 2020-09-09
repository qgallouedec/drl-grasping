"""Code for starting a training session.

Usage:
>>> mpirun -np 8 python main.py
"""

# The six following lines aims to ignore the numerous warnings of tensorflow. They can be removed.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from baselines.run import main

if __name__=='__main__':
    seed = 0
    main([
        '--num_env=1',
        '--alg=her',
        '--env=FetchPickAndPlace-v1',
        '--num_timesteps=200000',
        '--seed={}'.format(seed),
        '--save_path=policy_fetchpickandplace',
        '--log_path=~/logs/'
    ])
