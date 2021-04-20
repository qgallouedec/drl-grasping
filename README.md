# Baselines and panda-gym

Code to train an agent on [`panda-gym` environments](https://github.com/qgallouedec/panda-gym).

## Installation

Tested on Ubunutu 18.04 LTS.

1. install some dependencies

    ```shell
    sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
    ```

2. clone the repository

    ```shell
    git clone https://github.com/qgallouedec/drl_grasping
    ```

3. create a virtual environment, activate it and upgrade pip

    ```shell
    cd drl_grasping
    python3 -m venv env
    source env/bin/activate
    python -m pip install --upgrade pip
    ```

4. install dependecies

    ```shell
    pip install -r requirements.txt
    ```

## Usage

### Train

To train `PandaPickAndPlace-v1` with seed 0 for 500000 timesteps, run

```shell
mpirun -np 8 python train.py PandaPickAndPlace-v1 0 500000
```

The learning is distributed over 8 MPI workers. For the moment, this number should not be modified.

### Play

To play the learned policy, run

```shell
python play.py PandaPickAndPlace-v1
```

### Process the results

Turn the brut results into `.dat` file, containing timesteps, median and quartiles.

```shell
python results_to_dat.py
```

It process all the training done so far.


## Author

Quentin GALLOUÉDEC
