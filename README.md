# Deep Reinforcement Learning for Grasping



The work presented tries (but not succeed) to reproduces the results presented by [Plappert & al. (2018)](https://arxiv.org/abs/1802.09464). The idea is to use hinsight experience replay (HER) ([Andrychowicz & al. (2017)](http://papers.nips.cc/paper/7090-hindsight-experience-replay)) to learn four basic tasks.


## Installation

The following installation has been tested only on Ubuntu 18.04 LTS

1. install some dependencies

```shell
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

2. clone the repository

```shell
git clone https://github.com/quenting44/drl_grasping.git
```

3. move to this branch, create a virtual environment, activate it and upgrade pip

```shell
cd drl_grasping
git checkout stable-baselines
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
```

4. install tensorflow (or tensorflow-gpu if you have a CUDA-compatible gpu and proper drivers)

```shell
pip install tensorflow==1.14
# or
pip install tensorflow-gpu==1.14
```

5. install mujoco following the [instructions](https://github.com/openai/mujoco-py#install-mujoco)

6. install the remaining requirements.

```shell
pip install -r requirements.txt
```

## Usage

To train the Fetch robot to learn the pick and place task, run

```shell
mpirun -np 8 python main.py
```

The learning is distributed over 8 MPI workers and lasts 80 epochs. The learning data are stored under the `./log/` folder.

To turn the results into `.txt` files, you can run `python read.py`. It creates a table into a `.txt` for every configurations.

## Results

![](docs/results_four_envs.png)

## Author

Quentin GALLOUÉDEC
