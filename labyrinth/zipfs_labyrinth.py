# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A Zipfian distribution over DMLab-30 levels.

To use this environment, you will need to have DMLab installed. For
instructions, see:
https://github.com/deepmind/lab#getting-started-on-linux

For more information on the DMLab-30 levels, see the level definitions:
https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/dmlab30
"""
from typing import Sequence

import dm_env

import numpy as np

import deepmind_lab


LEVELS_DMLAB30_FORWARD = [
    'rooms_collect_good_objects_train', 'rooms_exploit_deferred_effects_train',
    'rooms_select_nonmatching_object', 'rooms_watermaze',
    'rooms_keys_doors_puzzle', 'language_select_described_object',
    'language_select_located_object', 'language_execute_random_task',
    'language_answer_quantitative_question', 'lasertag_one_opponent_small',
    'lasertag_three_opponents_small', 'lasertag_one_opponent_large',
    'lasertag_three_opponents_large', 'natlab_fixed_large_map',
    'natlab_varying_map_regrowth', 'natlab_varying_map_randomized',
    'skymaze_irreversible_path_hard', 'skymaze_irreversible_path_varied',
    'psychlab_arbitrary_visuomotor_mapping', 'psychlab_continuous_recognition',
    'psychlab_sequential_comparison', 'psychlab_visual_search',
    'explore_object_locations_small', 'explore_object_locations_large',
    'explore_obstructed_goals_small', 'explore_obstructed_goals_large',
    'explore_goal_locations_small', 'explore_goal_locations_large',
    'explore_object_rewards_few', 'explore_object_rewards_many'
]
# Reverse distribution
LEVELS_DMLAB30_REVERSED = list(reversed(LEVELS_DMLAB30_FORWARD))

DEFAULT_OBS = ('RGB_INTERLEAVED', 'INSTR')  # default observations


def zipf_dist(n, exponent=1.):
  vals = 1. / (np.arange(1, n + 1))**exponent
  return vals / np.sum(vals)


def uniform_rare_dist(n):
  vals = np.zeros([n], dtype=np.float32)
  num_rare = n // 5
  vals[-num_rare:] = 1. / num_rare
  return vals


def process_action_spec(lab_action_spec):
  """Processes action specs into dm_env format + produces action map."""
  action_spec = {}
  action_map = {}
  action_count = len(lab_action_spec)
  for i, spec in enumerate(lab_action_spec):
    name = spec['name']
    action_map[name] = i
    action_spec[name] = dm_env.specs.BoundedArray(
        dtype=np.dtype('int32'),
        shape=(),
        name=name,
        minimum=spec['min'],
        maximum=spec['max'])
  return action_spec, action_map, action_count


def process_observation_spec(lab_observation_spec, observation_names):
  """Filters observation specs and converts to dm_env format."""
  observation_spec = {}
  for spec in lab_observation_spec:
    name = spec['name']
    shape = spec['shape']
    if name in observation_names:
      observation_spec[name] = dm_env.specs.Array(
          dtype=spec['dtype'], shape=shape, name=name)
  return observation_spec


class ZipfsLabyrinth(dm_env.Environment):
  """Zipf's Labyrinth environment."""

  def __init__(self,
               distribution: str = 'zipf1',
               reverse_order: bool = False,
               observations_to_include: Sequence[str] = DEFAULT_OBS) -> None:
    """Create the environment.

    Args:
      distribution: name of probability distribution over levels; one of the
        following items ['uniform', 'uniform_rare', 'zipf1'].
      reverse_order: whether to reverse the order of the levels.
      observations_to_include: observations to keep from the environment,
        defaults to vision + instruction.
    """
    # set up distribution
    if distribution == 'uniform':
      self._distribution = np.ones((30,), dtype=np.float32) / 30
    elif distribution == 'uniform_rare':
      self._distribution = uniform_rare_dist(30)
    elif distribution[:4] == 'zipf':
      self._distribution = zipf_dist(30, exponent=float(distribution[4:]))

    # get level names
    if reverse_order:
      levels = LEVELS_DMLAB30_REVERSED
    else:
      levels = LEVELS_DMLAB30_FORWARD
    self.levels = ['contributed/dmlab30/' + l for l in levels]

    # build environments
    self._observations_to_include = observations_to_include
    self._environments = []
    observation_specs = []
    action_specs = []
    action_maps = []
    action_count = None
    for level in self.levels:
      env = deepmind_lab.Lab(level, ['RGB_INTERLEAVED', 'INSTR'])
      self._environments.append(env)
      # observation specs
      observation_spec = env.observation_spec()
      observation_spec = process_observation_spec(observation_spec,
                                                  observations_to_include)
      observation_specs.append(observation_spec)
      # action specs
      action_spec = env.action_spec()
      action_spec, action_map, action_count = process_action_spec(action_spec)
      action_specs.append(action_spec)
      action_maps.append(action_map)
    self._action_count = action_count

    assert all([spec == action_specs[0] for spec in action_specs[1:]])
    self._action_spec = action_specs[0]
    assert all([action_map == action_maps[0] for action_map in action_maps[1:]])
    self._action_map = action_maps[0]
    assert all([spec == observation_specs[0] for spec in observation_specs[1:]])
    self._observation_spec = observation_specs[0]

    self._current_env = None
    self._needs_reset = True

  def _sample_env(self):
    self._current_env = np.random.choice(self._environments,
                                         p=self._distribution)
    return self._current_env

  def _observation(self):
    return {
        name: np.asarray(data, dtype=self._observation_spec[name].dtype)
        for name, data in self._current_env.observations().items()
    }

  def reset(self):
    """Start a new episode."""
    # sample an environment
    self._sample_env()
    self._current_env.reset()
    self._needs_reset = False
    return dm_env.restart(self._observation())

  def step(self, action):
    if self._needs_reset:
      return self.reset()

    lab_action = np.empty(self._action_count, dtype=np.dtype('int32'))
    for name, value in action.items():
      lab_action[self._action_map[name]] = value

    reward = self._current_env.step(lab_action)

    if self._current_env.is_running():
      return dm_env.transition(reward=reward, observation=self._observation())
    else:
      self._needs_reset = True
      return dm_env.termination(reward=reward, observation=self._observation())

  @property
  def observation_spec(self):
    return self._observation_spec

  @property
  def action_spec(self):
    return self._action_spec
