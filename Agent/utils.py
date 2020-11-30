from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pysc2.lib import actions
from pysc2.lib import features
'''
height_map: Shows the terrain levels.
visibility: Which part of the map are hidden, have been seen or are currently visible.
creep: Which parts have zerg creep.
camera: Which part of the map are visible in the screen layers.
player_id: Who owns the units, with absolute ids.
player_relative: Which units are friendly vs hostile. Takes values in [0, 4], denoting [background, self, ally, neutral, enemy] units respectively.
selected: Which units are selected.
'''
# TODO: preprocessing functions for the following layers
_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index
_MINIMAP_UNIT_TYPE = features.MINIMAP_FEATURES.unit_type.index

_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
'''
Marine = 48
Zergling = 105
Roach = 110
Baneling = 9
SCV = 45
'''
unitList = [48,105,110,9,45]
def preprocess_minimap(minimap):
  layers = []
  assert minimap.shape[0] == len(features.MINIMAP_FEATURES)
  for i in range(len(features.MINIMAP_FEATURES)):
    if i == _MINIMAP_PLAYER_ID:
      layers.append(minimap[i:i+1] / features.MINIMAP_FEATURES[i].scale)
    elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
      layers.append(minimap[i:i+1] / features.MINIMAP_FEATURES[i].scale)
    elif i == _MINIMAP_UNIT_TYPE:
      '''
        unitList 是7个minigame中所有的unit种类，其他的不会在minigame中出现
      '''
      layer = np.zeros([ len(unitList) , minimap.shape[1], minimap.shape[2]], dtype=np.float32)
      for j in range(len(unitList)):
        indy, indx = (minimap[i] == unitList[j]).nonzero()
        layer[j, indy, indx] = 1
      layers.append(layer)
      #layers.append(minimap[i:i+1] / features.MINIMAP_FEATURES[i].scale)
    else:
      layer = np.zeros([features.MINIMAP_FEATURES[i].scale, minimap.shape[1], minimap.shape[2]], dtype=np.float32)
      for j in range(features.MINIMAP_FEATURES[i].scale):
        indy, indx = (minimap[i] == j).nonzero()
        layer[j, indy, indx] = 1
      layers.append(layer)
  return np.concatenate(layers, axis=0)


def preprocess_screen(screen):
  layers = []
  assert screen.shape[0] == len(features.SCREEN_FEATURES)
  for i in range(len(features.SCREEN_FEATURES)):
    if i == _SCREEN_PLAYER_ID:
      layers.append(screen[i:i+1] / features.SCREEN_FEATURES[i].scale)
    elif i == _SCREEN_UNIT_TYPE:
      '''
        unitList 是7个minigame中所有的unit种类，其他的不会在minigame中出现
      '''
      layer = np.zeros([ len(unitList) , screen.shape[1], screen.shape[2]], dtype=np.float32)
      for j in range(len(unitList)):
        indy, indx = (screen[i] == unitList[j]).nonzero()
        layer[j, indy, indx] = 1
      layers.append(layer)      
    elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
      layers.append(screen[i:i+1] / features.SCREEN_FEATURES[i].scale)
    else:
      layer = np.zeros([features.SCREEN_FEATURES[i].scale, screen.shape[1], screen.shape[2]], dtype=np.float32)
      for j in range(features.SCREEN_FEATURES[i].scale):
        indy, indx = (screen[i] == j).nonzero()
        layer[j, indy, indx] = 1
      layers.append(layer)
  return np.concatenate(layers, axis=0)


def minimap_channel():
  c = 0
  for i in range(len(features.MINIMAP_FEATURES)):
    if i == _MINIMAP_PLAYER_ID:
      c += 1
    elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
      c += 1
    elif i == _MINIMAP_UNIT_TYPE:
      c += len(unitList)
    else:
      c += features.MINIMAP_FEATURES[i].scale
  return c


def screen_channel():
  c = 0
  for i in range(len(features.SCREEN_FEATURES)):
    if i == _SCREEN_PLAYER_ID:
      c += 1
    elif i == _SCREEN_UNIT_TYPE:
      c += len(unitList)
    elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
      c += 1
    else:
      c += features.SCREEN_FEATURES[i].scale
  return c