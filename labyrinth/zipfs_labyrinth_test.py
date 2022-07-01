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

"""Tests for zipfs_gridworld."""
import os

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from zipfian_environments.labyrinth import zipfs_labyrinth as zipf_env
import deepmind_lab


class ZipfsLabyrinthTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="uniform",
           distribution="uniform", reverse_order=False),
      dict(testcase_name="uniform_rare",
           distribution="uniform_rare", reverse_order=False),
      dict(testcase_name="zipf_reversed",
           distribution="zipf1", reverse_order=True),
  )
  def test_labyrinth(self, distribution, reverse_order):
    env = zipf_env.ZipfsLabyrinth(
        distribution=distribution, reverse_order=reverse_order
    )
    for _ in range(3):
      timestep = env.reset()
      observation = timestep.observation
      self.assertEqual(set(observation.keys()),
                       set(["RGB_INTERLEAVED", "INSTR"]))
      # Instruction shuold be single string
      self.assertEqual(observation["INSTR"].shape, tuple())
      self.assertIsInstance(observation["INSTR"].item(), str)
      # vision should be RGB ints
      vision_shape = observation["RGB_INTERLEAVED"].shape
      self.assertLen(vision_shape, 3)
      self.assertEqual(vision_shape[-1], 3)
      self.assertEqual(observation["RGB_INTERLEAVED"].dtype, np.uint8)
      for i in [0, 1]:
        dummy_action = {k: i for k in env.action_spec.keys()}
        env.step(dummy_action)


if __name__ == "__main__":
  dmlab_runfiles_path = None  # set path here
  if dmlab_runfiles_path is not None:
    deepmind_lab.set_runfiles_path(dmlab_runfiles_path)
  absltest.main()
