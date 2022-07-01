## zipfian_environments

This repository contains the environments released as part of the paper "Zipfian
Environments for Reinforcement Learning" (https://arxiv.org/abs/2203.08222). It
contains subfolders for each of the three environments proposed in the paper:
1) Zipf's Playroom
2) Zipf's Labyrinth
3) Zipf's Gridworld

## Installation

### zipfs_labyrinth

Zipf's Labyrinth depends on DeepMind Labyrinth, which you will need to install
by following the instructions here: (https://github.com/deepmind/lab#getting-started-on-linux)

Once that is done, you can install the python requirements by running the
following commands:

```shell
python3 -m venv zipfian_environments
source zipfian_environments/bin/activate
pip install --upgrade pip
pip install -r labyrinth/requirements.txt
```

### zipfs_gridworld

To install the necessary requirements for Zipf's gridword, you can run the
following commands:

```shell
python3 -m venv zipfian_environments
source zipfian_environments/bin/activate
pip install --upgrade pip
pip install -r gridworld/requirements.txt
```

## Usage

### Zipf's Labyrinth

As an example, you can run the following code to load a Zipf's Labyrinth
environment.

```
from labyrinth import zipfs_labyrinth

env = zipfs_labyrinth.ZipfsLabyrinth(
  distribution="zipf1", reverse_order=False
)
timestep = env.reset()
```

You may first need to set the deepmind lab runfiles path as follows:

```
import deepmind_lab

deepmind_lab.set_runfiles_path(dmlab_runfiles_path)
```

### Zipf's Gridworld

As an example, you can run the following code to load a Zipf's Gridworld
environment and test out all the possible actions (note that the code will
need to be run from within this directory).


```
from gridworld import zipfs_gridworld

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
