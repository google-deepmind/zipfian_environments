## zipfian_environments

This repository contains the environments released as part of the paper "Zipfian
Environments for Reinforcement Learning" (https://arxiv.org/abs/2203.08222). It
contains subfolders for each of the three environments proposed in the paper:
1) Zipf's Playroom
2) Zipf's Labyrinth
3) Zipf's Gridworld

## Installation

### zipfs_playroom

These tasks are [Unity-based](http://unity3d.com/), and are provided through
pre-packaged [Docker containers](http://www.docker.com).

This package consists of support code to run these Docker containers. You
interact with the task environment via a
[`dm_env`](http://www.github.com/deepmind/dm_env) Python interface.

#### Requirements

This environment requires [Docker](https://www.docker.com),
[Python](https://www.python.org/) 3.6.1 or later and a x86-64 CPU with SSE4.2
support. We do not attempt to maintain a working version for Python 2.

Note: We recommend using
[Python virtual environment](https://docs.python.org/3/tutorial/venv.html) to
mitigate conflicts with your system's Python environment.

Download and install Docker:

*   For Linux, install [Docker-CE](https://docs.docker.com/install/)
*   Install Docker Desktop for
    [OSX](https://docs.docker.com/docker-for-mac/install/) or
    [Windows](https://docs.docker.com/docker-for-windows/install/).


#### Installation

To install `zipfs_playroom`:

```shell
python3 -m venv zipfs_playroom_venv
source zipfs_playroom_venv/bin/activate
pip install --upgrade pip
pip install -r ./zipfian_environments/playroom/requirements.txt
```

### zipfs_labyrinth

Zipf's Labyrinth depends on DeepMind Labyrinth, which you will need to install
by following the instructions [here](https://github.com/deepmind/lab/tree/master/python/pip_package).

Once that is done, you can install the python requirements by running the
following commands:

```shell
python3 -m venv zipfs_labyrinth_venv
source zipfs_labyrinth_venv/bin/activate
pip install --upgrade pip
pip install -r ./zipfian_environments/labyrinth/requirements.txt
```

### zipfs_gridworld

To install the necessary requirements for Zipf's gridword, you can run the
following commands:

```shell
python3 -m venv zipfs_gridworld_venv
source zipfs_gridworld_venv/bin/activate
pip install --upgrade pip
pip install -r ./zipfian_environments/gridworld/requirements.txt
```

## Usage

All examples assume execution from the directory that you cloned the
repository into.

### Zipf's Playroom

As an example, you can instantiate a `dm_env` instance by running the following:

```python
import zipfian_environments.playroom as zipfs_playroom

settings = zipfs_playroom.EnvironmentSettings(seed=123,
    level_name='lift/lift_shape_zipf2')
env = zipfs_playroom.load_from_docker(settings)
timestep = env.reset()
```

### Zipf's Labyrinth

As an example, you can run the following code to load a Zipf's Labyrinth
environment.

```
from zipfian_environments.labyrinth import zipfs_labyrinth

env = zipfs_labyrinth.ZipfsLabyrinth(
  distribution="zipf1", reverse_order=False
)
timestep = env.reset()
```

You may first need to set the deepmind lab runfiles path as follows:

```
import deepmind_lab

deepmind_lab.set_runfiles_path("INSERT PATH HERE")
```

### Zipf's Gridworld

As an example, you can run the following code to load a Zipf's Gridworld
environment and test out all the possible actions (note that the code will
need to be run from within this directory).


```
from zipfian_environments.gridworld import zipfs_gridworld

env = zipfs_gridworld.simple_builder(level_name='zipf_2')
timestep = env.reset()
for action in range(8):
  timestep = env.step(action)
```

## Citing this work

If you use this work, please cite the following paper
```
@misc{chan2022zipfian,
  doi = {10.48550/ARXIV.2203.08222},
  url = {https://arxiv.org/abs/2203.08222},
  author = {Chan, Stephanie C. Y. and Lampinen, Andrew K. and Richemond, Pierre H. and Hill, Felix},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Zipfian environments for Reinforcement Learning},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Disclaimer

This is not an official Google product.
