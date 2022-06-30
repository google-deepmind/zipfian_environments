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

"""The pycolab core of the environment for Zipf's Gridworld."""
import curses
import enum

from absl import app
from absl import flags
from absl import logging

import numpy as np

from pycolab import ascii_art
from pycolab import human_ui
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites

from zipfian_environments.gridworld import map_building


FLAGS = flags.FLAGS

DEFAULT_NUM_ROOMS = (3, 3)
DEFAULT_ROOM_SIZE = (3, 3)  # note one square around each room will be wall.
DEFAULT_NUM_MAPS = 20
DEFAULT_NUM_OBJECTS = 20
DEFAULT_ZIPF_EXPONENT = 2


AGENT_CHAR = map_building.AGENT_CHAR
WALL_CHAR = map_building.WALL_CHAR
FLOOR_CHAR = map_building.FLOOR_CHAR


OBJECT_SHAPES = [
    "triangle", "empty_square", "plus", "inverse_plus", "ex", "inverse_ex",
    "circle", "empty_circle", "tee", "upside_down_tee",
    "h", "u", "upside_down_u", "vertical_stripes", "horizontal_stripes"
]

COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([128, 0, 128]),
    "orange": np.array([255, 165, 0]),
    "yellow": np.array([255, 255, 0]),
    "brown": np.array([128, 64, 0]),
    "pink": np.array([255, 64, 255]),
    "cyan": np.array([0, 255, 255]),
    "dark_green": np.array([0, 100, 0]),
    "dark_red": np.array([100, 0, 0]),
    "dark_blue": np.array([0, 0, 100]),
    "teal": np.array([0, 100, 100]),
    "lavender": np.array([215, 200, 255]),
    "rose": np.array([255, 205, 230]),
}


def zipf_dist(n, exponent=1.):
  vals = 1./(np.arange(1, n + 1))**exponent
  return vals / np.sum(vals)


def uniform_rare_dist(n):
  vals = np.zeros([n], dtype=np.float32)
  num_rare = n // 5
  vals[-num_rare:] = 1. / num_rare
  return vals


# movement directions / actions for agent
class DIRECTIONS(enum.IntEnum):
  N = 0
  NE = 1
  E = 2
  SE = 3
  S = 4
  SW = 5
  W = 6
  NW = 7


class ObjectDrape(plab_things.Drape):
  """A `Drape` for objects that are to be found."""

  def __init__(self, curtain, character, color, shape, value=0.):
    super(ObjectDrape, self).__init__(curtain, character)
    self.color = color
    self.shape = shape
    self.value = value

  def update(self, actions, board, layers, backdrop, things, the_plot):
    if self.curtain[things[AGENT_CHAR].position]:
      # Award the player the appropriate amount of reward.
      the_plot.add_reward(self.value)
      the_plot.terminate_episode()


class PlayerSprite(prefab_sprites.MazeWalker):
  """The player / agent character."""

  def __init__(self, corner, position, character):
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable="#")

  def update(self, actions, board, layers, backdrop, things, the_plot):
    # agent can move around to look for the target
    if actions == DIRECTIONS.N:
      self._north(board, the_plot)
    elif actions == DIRECTIONS.NE:
      self._northeast(board, the_plot)
    elif actions == DIRECTIONS.E:
      self._east(board, the_plot)
    elif actions == DIRECTIONS.SE:
      self._southeast(board, the_plot)
    elif actions == DIRECTIONS.S:
      self._south(board, the_plot)
    elif actions == DIRECTIONS.SW:
      self._southwest(board, the_plot)
    elif actions == DIRECTIONS.W:
      self._west(board, the_plot)
    elif actions == DIRECTIONS.NW:
      self._northwest(board, the_plot)


def make_game(house_map, objects_and_properties):
  """Uses pycolab to turn an ascii map into a game.

  Args:
    house_map: the ascii art map to use, from generate_map.
    objects_and_properties: list of (character, color, shape, value), for
      setting up object properties.

  Returns:
    this_game: Pycolab engine running the specified game.
  """
  sprites = {AGENT_CHAR: PlayerSprite}
  drape_creators = {}
  char_to_color_shape = []

  # add dancers to level
  for obj, color, shape, value in objects_and_properties:
    drape_creators[obj] = ascii_art.Partial(
        ObjectDrape, color=color, shape=shape, value=value)
    char_to_color_shape.append((obj, color + " " + shape))
    if value > 0.:
      target_color_shape = color + " " + shape

  this_game = ascii_art.ascii_art_to_game(
      art=house_map,
      what_lies_beneath=" ",
      sprites=sprites,
      drapes=drape_creators,
      update_schedule=[[AGENT_CHAR],
                       [x[0] for x in objects_and_properties]])

  this_game.the_plot["target_color_shape"] = target_color_shape
  this_game.the_plot["char_to_color_shape"] = char_to_color_shape
  return this_game


def builder(
    num_maps=DEFAULT_NUM_MAPS,
    num_rooms=DEFAULT_NUM_ROOMS,
    room_size=DEFAULT_ROOM_SIZE,
    num_objects=DEFAULT_NUM_OBJECTS,
    zipf_exponent=DEFAULT_ZIPF_EXPONENT,
    override_dist_function=None,
    rng=None):
  """Builds a game (a single episode) of Zipf's Gridworld.

  Args:
    num_maps: The number of maps to sample from.
    num_rooms: (n, m) to create a n x m grid of rooms.
    room_size: (k, l) to create a k x l room shape.
    num_objects: The number of objects to include per map.
    zipf_exponent: the exponent to use for the Zipfian distribution, or 0 for
      uniform.
    override_dist_function: A function that takes as input a number of items,
      and outputs a vector giving distribution over that number of items. This
      is used for custom evaluation, e.g. on only the rare items.
    rng: Optional np rng for reproducible results, e.g.
      rng=np.random.default_rng(seed=0).

  Returns:
    A pycolab game sampled according to the specified distributions.
  """
  if override_dist_function is None:
    dist_function = lambda n: zipf_dist(n, exponent=zipf_exponent)
  else:
    dist_function = override_dist_function
  # set up distribution over and sample index
  map_weighting = dist_function(num_maps)
  if rng is None:
    rng = np.random.default_rng(1234)
  map_index = rng.choice(np.arange(num_maps), p=map_weighting)
  # always same map for given index
  map_rng = np.random.default_rng(seed=1234 + map_index)
  current_map, objects_in_map = map_building.generate_map(
      n_rooms=num_rooms, room_size=room_size, n_objs=num_objects,
      rng=map_rng
  )

  # set up object weighting and sample target object
  object_weighting = dist_function(num_objects)
  target_index = rng.choice(np.arange(num_objects),
                            p=object_weighting)
  values = [1. if i == target_index else 0. for i in range(num_objects)]

  # generate set of colors and shapes such that all pairings are unique
  these_colors = []
  these_shapes = []
  while len(these_colors) < num_objects or len(these_shapes) < num_objects:
    these_colors = map_rng.choice(list(COLORS.keys()), num_objects + 10,
                                  replace=True)
    these_shapes = map_rng.choice(OBJECT_SHAPES[:], num_objects + 10,
                                  replace=True)
    # ensure unique pairings
    these_colors_and_shapes = sorted(list(set(zip(these_colors, these_shapes))))
    map_rng.shuffle(these_colors_and_shapes)
    these_colors, these_shapes = zip(*these_colors_and_shapes)

  objects_and_properties = list(zip(objects_in_map, these_colors,
                                    these_shapes, values))
  logging.info(
      "Making game with map_index %i and objects %s",
      map_index, str(objects_and_properties))
  game = make_game(current_map, objects_and_properties)
  return game


def main(argv):
  del argv  # unused

  game = builder()

  # Note that these colors are only for human UI
  fg_colors = {
      AGENT_CHAR: (999, 999, 999),  # Agent is white
      WALL_CHAR: (300, 300, 300),  # Wall, dark grey
      FLOOR_CHAR: (0, 0, 0),  # Floor
  }

  obj_chars = map_building.POSSIBLE_OBJECT_CHARS[:20]
  cols = [(999 * x / 255.).astype(np.int32)for x in COLORS.values()]
  cols = [tuple(x.tolist()) for x in cols] * 2
  obj_chars_and_colors = zip(
      obj_chars, cols
  )
  fg_colors.update(dict(obj_chars_and_colors))

  bg_colors = {
      c: (0, 0, 0) for c in map_building.RESERVED_CHARS + obj_chars
  }

  ui = human_ui.CursesUi(
      keys_to_actions={
          # Basic movement.
          curses.KEY_UP: DIRECTIONS.N,
          curses.KEY_DOWN: DIRECTIONS.S,
          curses.KEY_LEFT: DIRECTIONS.W,
          curses.KEY_RIGHT: DIRECTIONS.E,
          "w": DIRECTIONS.N,
          "d": DIRECTIONS.E,
          "x": DIRECTIONS.S,
          "a": DIRECTIONS.W,
          "q": DIRECTIONS.NW,
          "e": DIRECTIONS.NE,
          "z": DIRECTIONS.SW,
          "c": DIRECTIONS.SE,
          -1: 8,  # Do nothing.
      },
      delay=500,
      colour_fg=fg_colors,
      colour_bg=bg_colors)

  ui.play(game)


if __name__ == "__main__":
  app.run(main)
