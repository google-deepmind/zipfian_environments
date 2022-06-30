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

"""A pycolab environment for finding objects in skewed environments.

Both the maps and the objects to find in each map are zipfian distributed.

The room is upsampled at a size of 9 pixels per square to render a view for the
agent, which is cropped in egocentric perspective, i.e. the agent is always in
the center of its view (see https://arxiv.org/abs/1910.00571).
"""
from absl import flags

import dm_env

import numpy as np
from pycolab import cropping

from zipfian_environments.gridworld import zipfs_gridworld_core as env_core


FLAGS = flags.FLAGS

UPSAMPLE_SIZE = 9  # pixels per game square
SCROLL_CROP_SIZE = 7  # in game squares

OBJECT_SHAPES = env_core.OBJECT_SHAPES
COLORS = env_core.COLORS


def _generate_template(object_name):
  """Generates a template object image, given a name with color and shape."""
  object_color, object_type = object_name.split()
  template = np.zeros((UPSAMPLE_SIZE, UPSAMPLE_SIZE))
  half = UPSAMPLE_SIZE // 2
  if object_type == "triangle":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if (j <= half and i >= 2 * (half - j)) or (j > half and i >= 2 *
                                                   (j - half)):
          template[i, j] = 1.
  elif object_type == "square":
    template[:, :] = 1.
  elif object_type == "empty_square":
    template[:2, :] = 1.
    template[-2:, :] = 1.
    template[:, :2] = 1.
    template[:, -2:] = 1.
  elif object_type == "plus":
    template[:, half - 1:half + 2] = 1.
    template[half - 1:half + 2, :] = 1.
  elif object_type == "inverse_plus":
    template[:, :] = 1.
    template[:, half - 1:half + 2] = 0.
    template[half - 1:half + 2, :] = 0.
  elif object_type == "ex":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if abs(i - j) <= 1 or abs(UPSAMPLE_SIZE - 1 - j - i) <= 1:
          template[i, j] = 1.
  elif object_type == "inverse_ex":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if not (abs(i - j) <= 1 or abs(UPSAMPLE_SIZE - 1 - j - i) <= 1):
          template[i, j] = 1.
  elif object_type == "circle":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if (i - half)**2 + (j - half)**2 <= half**2:
          template[i, j] = 1.
  elif object_type == "empty_circle":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if abs((i - half)**2 + (j - half)**2 - half**2) < 6:
          template[i, j] = 1.
  elif object_type == "tee":
    template[:, half - 1:half + 2] = 1.
    template[:3, :] = 1.
  elif object_type == "upside_down_tee":
    template[:, half - 1:half + 2] = 1.
    template[-3:, :] = 1.
  elif object_type == "h":
    template[:, :3] = 1.
    template[:, -3:] = 1.
    template[half - 1:half + 2, :] = 1.
  elif object_type == "u":
    template[:, :3] = 1.
    template[:, -3:] = 1.
    template[-3:, :] = 1.
  elif object_type == "upside_down_u":
    template[:, :3] = 1.
    template[:, -3:] = 1.
    template[:3, :] = 1.
  elif object_type == "vertical_stripes":
    for j in range(half + UPSAMPLE_SIZE % 2):
      template[:, 2*j] = 1.
  elif object_type == "horizontal_stripes":
    for i in range(half + UPSAMPLE_SIZE % 2):
      template[2*i, :] = 1.
  else:
    raise ValueError("Unknown object: {}".format(object_type))

  if object_color not in COLORS:
    raise ValueError("Unknown color: {}".format(object_color))

  template = np.tensordot(template, COLORS[object_color], axes=0)

  return template


# Agent and wall templates
_CHAR_TO_TEMPLATE_BASE = {
    env_core.AGENT_CHAR:
        np.tensordot(
            np.ones([UPSAMPLE_SIZE, UPSAMPLE_SIZE]),
            np.array([255, 255, 255]),
            axes=0),
    env_core.WALL_CHAR:
        np.tensordot(
            np.ones([UPSAMPLE_SIZE, UPSAMPLE_SIZE]),
            np.array([40, 40, 40]),
            axes=0),
}

# cue background color is light gray
CUE_BACKGROUND = np.array([180, 180, 180])


def add_background_color_to_image(image, background_color):
  """Replaces zeros in the image with background color."""
  return np.where(
      np.all(image == 0., axis=-1, keepdims=True),
      background_color[None, None, :],
      image
  )


def get_scrolling_cropper(rows=SCROLL_CROP_SIZE, cols=SCROLL_CROP_SIZE,
                          crop_pad_char=env_core.FLOOR_CHAR):
  """Sets up cropper for egocentric agent view."""
  return cropping.ScrollingCropper(rows=rows, cols=cols,
                                   to_track=[env_core.AGENT_CHAR],
                                   pad_char=crop_pad_char,
                                   scroll_margins=(None, None))


class ZipfsGridworldEnvironment(dm_env.Environment):
  """A Python environment API for finding tasks."""

  def __init__(
      self,
      num_maps=env_core.DEFAULT_NUM_MAPS,
      num_rooms=env_core.DEFAULT_NUM_ROOMS,
      room_size=env_core.DEFAULT_ROOM_SIZE,
      num_objects=env_core.DEFAULT_NUM_OBJECTS,
      zipf_exponent=env_core.DEFAULT_ZIPF_EXPONENT,
      override_dist_function=None,
      max_steps=100,
      rng=None):
    """Construct a dm_env-compatible wrapper for pycolab games for agent use.

    This class inherits from dm_env and has all the expected methods and specs.
    It also renders the game from ascii art to pixel observations.

    Args:
      num_maps: The number of possible maps to sample from.
      num_rooms: The number of rooms per map, as a tuple of
        (num_rooms_x, num_rooms_y); rooms are layed out in a grid of this size.
      room_size: Tuple of (size_x, size_y) for each room, measured in game
        squares.
      num_objects: The number of objects to place in each map.
      zipf_exponent: The strength of the skew, use 0 for a uniform distribution.
      override_dist_function: Function that generates a distribution, in lieu of
        the zipf_distribution function.
      max_steps: The maximum number of steps to allow in an episode, after which
          it will terminate.
      rng: An optional numpy Random Generator, to set a fixed seed use e.g.
          `rng=np.random.default_rng(seed=...)`
    """
    # args
    self._builder_kwargs = dict(
        num_maps=num_maps,
        num_rooms=num_rooms,
        room_size=room_size,
        num_objects=num_objects,
        zipf_exponent=zipf_exponent,
        override_dist_function=override_dist_function
    )

    self._max_steps = max_steps

    # internal state
    if rng is None:
      rng = np.random.default_rng(seed=1234)
    self._rng = rng
    self._current_game = None       # Current pycolab game instance.
    self._state = None              # Current game step state.
    self._game_over = None          # Whether the game has ended.
    self._char_to_template = None   # Mapping of chars to images.
    self._cue_template = None

    # rendering tools
    self._cropper = get_scrolling_cropper(SCROLL_CROP_SIZE, SCROLL_CROP_SIZE,
                                          env_core.FLOOR_CHAR)

  def _render_observation(self, observation):
    """Renders from raw pycolab image observation to agent-usable pixel ones."""
    observation = self._cropper.crop(observation)
    obs_rows, obs_cols = observation.board.shape
    image = np.zeros([obs_rows * UPSAMPLE_SIZE, obs_cols * UPSAMPLE_SIZE, 3],
                     dtype=np.float32)
    for i in range(obs_rows):
      for j in range(obs_cols):
        this_char = chr(observation.board[i, j])
        if this_char != env_core.FLOOR_CHAR:
          image[
              i * UPSAMPLE_SIZE:(i + 1) * UPSAMPLE_SIZE, j *
              UPSAMPLE_SIZE:(j + 1) * UPSAMPLE_SIZE] = self._char_to_template[
                  this_char]

    # insert cue image, always in agent's top left corner
    image[:UPSAMPLE_SIZE, :UPSAMPLE_SIZE] = self._cue_template
    image /= 255.
    return (image,)

  def reset(self):
    """Start a new episode."""
    # Build a new game and retrieve its first set of state/reward/discount.
    self._current_game = env_core.builder(
        rng=self._rng,
        **self._builder_kwargs)
    # set up rendering, cropping, and state for current game
    self._char_to_template = {
        k: _generate_template(v) for k, v in self._current_game.the_plot[
            "char_to_color_shape"]}
    self._char_to_template.update(_CHAR_TO_TEMPLATE_BASE)
    self._cue_template = add_background_color_to_image(
        _generate_template(
            self._current_game.the_plot["target_color_shape"]),
        CUE_BACKGROUND)
    self._cropper.set_engine(self._current_game)
    self._state = dm_env.StepType.FIRST
    # let's go!
    observation, _, _ = self._current_game.its_showtime()
    observation = self._render_observation(observation)
    return dm_env.TimeStep(
        step_type=self._state,
        reward=None,
        discount=None,
        observation=observation)

  def step(self, action):
    """Apply action, step the world forward, and return observations."""
    # If needed, reset and start new episode.
    if self._state == dm_env.StepType.LAST:
      self._clear_state()
    if self._current_game is None:
      return self.reset()

    # Execute the action in pycolab.
    observation, reward, discount = self._current_game.play(action)

    self._game_over = self._is_game_over()
    reward = reward if reward is not None else 0.
    observation = self._render_observation(observation)

    # Check the current status of the game.
    if self._game_over:
      self._state = dm_env.StepType.LAST
    else:
      self._state = dm_env.StepType.MID

    return dm_env.TimeStep(
        step_type=self._state,
        reward=reward,
        discount=discount,
        observation=observation)

  def observation_spec(self):
    image_shape = (SCROLL_CROP_SIZE * UPSAMPLE_SIZE,
                   SCROLL_CROP_SIZE * UPSAMPLE_SIZE,
                   3)
    return (
        # vision
        dm_env.specs.Array(
            shape=image_shape, dtype=np.float32, name="image"),
        )

  def action_spec(self):
    return dm_env.specs.BoundedArray(
        shape=[], dtype="int32",
        minimum=0, maximum=7,
        name="grid_actions")

  def _is_game_over(self):
    """Returns whether it is game over, either from the engine or timeout."""
    return (self._current_game.game_over or
            (self._current_game.the_plot.frame >= self._max_steps))

  def _clear_state(self):
    """Clear all the internal information about the game."""
    self._state = None
    self._current_game = None
    self._char_to_template = None
    self._game_over = None


def simple_builder(level_name, level_kwargs_overrides=None):
  """Simplifies building from fixed defs.

  Args:
    level_name: one of 'uniform', 'uniform_rare', or 'zipf_{exponent}' for
      {exponent} interpretable as a float.
    level_kwargs_overrides: optional dict of overrides to default level
      settings, e.g. to run on larger maps or with more objects.

  Returns:
    A ZipfsGridworldEnvironment with the requested settings.
  """
  level_args = dict(
      num_maps=env_core.DEFAULT_NUM_MAPS,
      num_rooms=env_core.DEFAULT_NUM_ROOMS,
      room_size=env_core.DEFAULT_ROOM_SIZE,
      num_objects=env_core.DEFAULT_NUM_OBJECTS,
  )
  if level_kwargs_overrides is not None:
    level_args.update(level_kwargs_overrides)
  if level_name == "uniform":
    level_args["zipf_exponent"] = 0
  elif level_name == "uniform_rare":
    level_args["zipf_exponent"] = 0
    level_args["override_dist_function"] = env_core.uniform_rare_dist
  elif level_name[:4] == "zipf":
    level_args["zipf_exponent"] = float(level_name[5:])
  else:
    raise ValueError("Unrecognized level name: " + level_name)

  return ZipfsGridworldEnvironment(**level_args)
