"""Convert results into .dat files, headed by
timestep med lowq highq
"""


import csv
import os

import numpy as np


def process(env_id):
    success_rates = []
    seed = 0
    while True:
        success_rate = []
        file_name = "results/{}/{}/progress.csv".format(env_id, seed)
        # check if the filename exist, if not, break
        if not os.path.exists(file_name):
            break
        print(file_name)
        with open(file_name, "r") as f:
            reader = csv.reader(f)
            next(reader)  # ignore heading
            for row in reader:
                success_rate.append(row[7])  # row 7 is the test sucess rate
        success_rates.append(success_rate)
        seed += 1

    if not success_rates:
        return
    success_rates = np.array(success_rates, dtype=np.float)
    med = np.median(success_rates, axis=0)
    l = len(med)
    episode_length = 10 if env_id == "PandaStack-v1" else 50
    n_cycles = 10 if env_id == "PandaReach-v1" else 50
    n_mpi_workers = 8
    timesteps = np.arange(1, l + 1) * episode_length * n_cycles * n_mpi_workers
    lowq = np.quantile(success_rates, 0.33, axis=0)
    highq = np.quantile(success_rates, 0.66, axis=0)
    out = np.vstack((timesteps, med, lowq, highq)).transpose()

    np.savetxt(
        "results/{}.dat".format(env_id),
        out,
        fmt="%.3f",
        header="timestep med lowq highq",
        comments="",
    )


if __name__ == "__main__":
    for env_id in [
        "PandaReach-v1",
        "PandaPush-v1",
        "PandaSlide-v1",
        "PandaPickAndPlace-v1",
    ]:
        process(env_id)
