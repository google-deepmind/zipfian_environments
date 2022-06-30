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

"""Utilities for building ascii maps with multiple rooms."""
import itertools
import numpy as np


AGENT_CHAR = "A"
WALL_CHAR = "#"
FLOOR_CHAR = " "
RESERVED_CHARS = [AGENT_CHAR, WALL_CHAR, FLOOR_CHAR]
POSSIBLE_OBJECT_CHARS = [
    chr(i) for i in range(65, 91) if chr(i) not in RESERVED_CHARS
]


def manhattan_dist(x, y):
  return abs(x[0] - y[0]) + abs(x[1] - y[1])


def get_neighbors(room, rooms_list):
  return [other for other in rooms_list if manhattan_dist(room, other) == 1]


def ensure_connected(rooms_list, room_doors, rng):
  """Adds doors until all rooms are connected to each other."""
  connected = rooms_list[:1]
  unconnected = []
  uncertain = rooms_list[1:]
  while uncertain or unconnected:
    if uncertain:
      connected_new = []
      for room in uncertain:
        for other in connected:
          if room in room_doors[other]:
            connected.append(room)
            connected_new.append(room)
            break
      if connected_new:  # remove from uncertain
        uncertain = list(set(uncertain) - set(connected_new))
      else:  # nothing new, remaining are unconnected
        unconnected = uncertain
        uncertain = []
    elif unconnected:
      connected_new = []
      rng.shuffle(unconnected)
      for room in unconnected:
        connected_neighbors = get_neighbors(room, connected)
        if connected_neighbors:
          rng.shuffle(connected_neighbors)
          other = connected_neighbors[0]
          room_doors[room].append(other)
          room_doors[other].append(room)
          connected_new.append(room)
          connected.append(room)
          break  # otherwise we may overdo it on connections
      assert connected_new  # should be at least 1 adjacency
      uncertain = list(set(unconnected) - set(connected_new))
      unconnected = []
  return room_doors


def generate_map(n_rooms=(3, 3), room_size=(3, 3), n_objs=20, rng=None):
  """Generates a map for Zipf's Gridworld."""
  if rng is None:
    rng = np.random.default_rng(seed=0)
  rooms_list = list(itertools.product(range(n_rooms[0]), range(n_rooms[1])))
  room_doors = {room: [] for room in rooms_list}
  # start by adding a random door to each room
  for room in rooms_list:
    if not room_doors[room]:  # if it doesn't already have a door
      neighbors = get_neighbors(room, rooms_list)
      other = neighbors[rng.integers(len(neighbors))]
      room_doors[room].append(other)
      room_doors[other].append(room)
  # now make sure it's connected
  room_doors = ensure_connected(rooms_list, room_doors, rng)
  # generate level
  row_skip = room_size[0] + 1  # extra for walls
  col_skip = room_size[1] + 1
  num_rows = n_rooms[0] * row_skip + 1
  num_cols = n_rooms[1] * col_skip + 1
  level_layout = np.array([[" "] * num_cols] * num_rows)
  # insert walls
  for i in range(n_rooms[0] + 1):
    level_layout[i * row_skip, :] = WALL_CHAR
  for i in range(n_rooms[1] + 1):
    level_layout[:, i * col_skip] = WALL_CHAR
  # add doors
  alldoors = []
  for room in rooms_list:
    for other in set(room_doors[room]):
      if other[0] < room[0]:
        loc = (room[0] * row_skip,
               room[1] * col_skip + rng.integers(1, col_skip))
        level_layout[loc] = FLOOR_CHAR
        alldoors.append(loc)
      elif other[1] < room[1]:
        loc = (room[0] * row_skip + rng.integers(1, row_skip),
               room[1] * col_skip)
        level_layout[loc] = FLOOR_CHAR
        alldoors.append(loc)
  # add objects (and agent start) where they will not block anything
  valid_locations = []
  for i in range(num_rows):
    for j in range(num_cols):
      this_loc = (i, j)
      if (i % row_skip) == 0 or (j % col_skip) == 0:
        continue  # inside wall
      if not (((i % row_skip) in (1, row_skip - 1))
              or ((j % col_skip) in (1, col_skip - 1))):
        continue  # not against a wall
      adjacencies = [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)]
      if any(x in alldoors for x in adjacencies):
        continue  # blocking a door
      valid_locations.append(this_loc)
  objs = POSSIBLE_OBJECT_CHARS[:n_objs] + [AGENT_CHAR]
  assert len(valid_locations) > len(objs)
  rng.shuffle(valid_locations)
  for obj, loc in zip(objs, valid_locations):
    level_layout[loc] = obj
  objs_to_drape = objs[:-1]
  # convert to pycolab's ascii format
  level_layout = ["".join(x) for x in level_layout.tolist()]
  return level_layout, objs_to_drape

