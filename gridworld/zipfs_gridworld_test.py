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
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from zipfian_environments.gridworld import zipfs_gridworld as zipf_env
from zipfian_environments.gridworld import zipfs_gridworld_core as zipf_env_core


class ZipfFindEnvironmentTest(parameterized.TestCase):

  def test_full_wrapper(self):
    env = zipf_env.ZipfsGridworldEnvironment(
        rng=np.random.default_rng(seed=0)
    )
    result = env.reset()
    self.assertIsNone(result.reward)
    scroll_crop_size = zipf_env.SCROLL_CROP_SIZE
    upsample_size = zipf_env.UPSAMPLE_SIZE
    self.assertEqual(result.observation[0].shape,
                     (scroll_crop_size * upsample_size,
                      scroll_crop_size * upsample_size,
                      3))
    us = zipf_env.UPSAMPLE_SIZE
    scs = zipf_env.SCROLL_CROP_SIZE
    offset = us * (scs // 2)
    for i in range(8):  # check all actions work
      result = env.step(i)
      self.assertEqual(result.observation[0].shape,
                       (scroll_crop_size * upsample_size,
                        scroll_crop_size * upsample_size,
                        3))
      if not result.step_type.last:
        # check egocentric scrolling is working, by checking agent is in center
        np.testing.assert_array_almost_equal(
            result.observation[0][offset:offset + us, offset:offset + us, :],
            zipf_env._CHAR_TO_TEMPLATE_BASE[zipf_env_core.AGENT_CHAR] / 255.)
      # check goal object cue in the top left as intended
      np.testing.assert_array_almost_equal(
          result.observation[0][:us, :us, :],
          env._cue_template / 255.)

  @parameterized.parameters(
      "uniform",
      "uniform_rare",
      "zipf_1",
      "zipf_1.5",
      "zipf_2",
  )
  def test_simple_builder(self, level_name):
    env = zipf_env.simple_builder(level_name)
    env.reset()
    scroll_crop_size = zipf_env.SCROLL_CROP_SIZE
    upsample_size = zipf_env.UPSAMPLE_SIZE
    for i in range(8):
      result = env.step(i)  # check all 8 movements work
      self.assertEqual(result.observation[0].shape,
                       (scroll_crop_size * upsample_size,
                        scroll_crop_size * upsample_size,
                        3))

if __name__ == "__main__":
  absltest.main()
