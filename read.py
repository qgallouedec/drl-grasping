"""Turn training output file into .txt file (in order to be used in LaTeX)
"""

import numpy as np
import os
from pathlib import Path


def list_array_equal(list_arrays):
    """Whether all array in the list are equal"""
    assert len(list_arrays) > 0
    if len(list_arrays) == 1:
        return True
    if len(list_arrays) == 2:
        return np.array_equal(*list_arrays)
    else:
        return list_array_equal(list_arrays[1:]) and list_array_equal(
            list_arrays[:2])



def do(exp_path):
    #log/exp_name/env_name/method/reward_type/training_id/evaluations.zip

    # get a list of paths of the training output files
    list_paths = [
        directory.joinpath('evaluations.npz')
        for directory in exp_path.iterdir()
    ]

    # remove wrong paths (that does not exist)
    list_paths = [path for path in list_paths if os.path.exists(path)]

    # get the dict contained in the files
    # it contains threee keys : results, timesteps and I forgot the last one (TODO)
    list_evaluations = [dict(np.load(path)) for path in list_paths]

    # get an array of corresponding timesteps and check
    # that all training has the same list of timesteps
    list_timesteps = [evaluation['timesteps'] for evaluation in list_evaluations]
    assert list_array_equal(list_timesteps), "Timesteps aren't matching"

    timesteps = list_timesteps[0]

    # get the list of rewards for each training
    # each item of the following list is a array where axis 0 is the timestep
    # and axis 1 is the idx of the evaluation
    list_results = [
        evaluation['results'].squeeze() for evaluation in list_evaluations
    ]

    # turn reward into success ({0, 1}). If the reward is > -50, it means
    # that the robotic arm succeed
    list_successes = [(result > -50).astype(int) for result in list_results]

    # for each timestep, compute the average success rate among all evaluations
    list_success_rates = [np.mean(success, axis=1) for success in list_successes]

    # stack the success rates to get a array where 
    # axis 0 is the idx of training and axis 1 the timestep
    success_rates = np.stack(list_success_rates, axis=0)

    # compute moments
    success_rate_mean = np.mean(success_rates, axis=0)
    success_rate_std = np.std(success_rates, axis=0)
    success_rate_med = np.quantile(success_rates, 0.5, axis=0)
    success_rate_quantile_low = np.quantile(success_rates, 0.25, axis=0)
    success_rate_quantile_high = np.quantile(success_rates, 0.75, axis=0)

    # stack moments together
    success_rate_moments = np.stack(
        (timesteps, success_rate_mean, success_rate_std, success_rate_med,
        success_rate_quantile_low, success_rate_quantile_high),
        axis=1)

    return success_rate_moments


if __name__=='__main__':
    # log match the following pattern
    # log/exp_name/env_name/method/reward_type/training_id/evaluations.zip
    exp_name = 'my_exp_name'
    cwd = os.getcwd()
    base_path = Path(cwd)
    log_path = base_path.joinpath('log')
    assert os.path.exists(log_path)
    exp_path = log_path.joinpath(exp_name)
    assert os.path.exists(exp_path)
    for env_path in exp_path.iterdir():
        env_name = os.path.basename(env_path)
        if not env_path.is_dir():
            continue
        for method_path in env_path.iterdir():
            method_name = os.path.basename(method_path)
            for reward_type_path in method_path.iterdir():
                reward_type_name = os.path.basename(reward_type_path)
                exp_moments = do(reward_type_path)
                np.savetxt(exp_path.joinpath('{}_{}_{}.txt'.format(env_name, method_name, reward_type_name)),
                    exp_moments,
                    fmt='%d %.5f %.5f %.5f %.5f %.5f',
                    header='timestep mean std med lowq highq',
                    comments='')
