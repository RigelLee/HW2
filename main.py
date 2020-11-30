#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run an agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import threading
import os
import sys
import tensorflow as tf

from absl import app
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch

from run_loop import run_loop
from Agent.agent import A3CAgent
COUNTER = 0
LOCK = threading.Lock()

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
point_flag.DEFINE_point("feature_screen_size", "64",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", None,
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", None,
                        "Resolution for rendered minimap.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")
flags.DEFINE_bool("use_raw_units", False,
                  "Whether to include raw units.")

flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("continuation", True, "Continuously training.")
flags.DEFINE_integer("max_agent_steps", 60, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("max_steps", int(1e6), "Total steps for training.")
flags.DEFINE_integer("snapshot_step", int(500), "Step for snapshot.")
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_string("log_path", "./log/", "Path for log.")
flags.DEFINE_string("device", "0", "Device for training.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 8, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
#flags.mark_flag_as_required("map")

FLAGS(sys.argv)
if FLAGS.training:
  PARALLEL = FLAGS.parallel
  MAX_AGENT_STEPS = FLAGS.max_agent_steps
  DEVICE = ['/gpu:'+dev for dev in FLAGS.device.split(',')]
else:
  PARALLEL = 1
  MAX_AGENT_STEPS = 1e5
  DEVICE = ['/cpu:0']

LOG = FLAGS.log_path+FLAGS.map
SNAPSHOT = FLAGS.snapshot_path+FLAGS.map
if not os.path.exists(LOG):
  os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
  os.makedirs(SNAPSHOT)


def run_thread(agent, players, map_name, visualize):
  """Run one thread worth of the environment with agents."""
  with sc2_env.SC2Env(
      map_name=map_name,
      players=players,
      agent_interface_format=sc2_env.parse_agent_interface_format(
        feature_screen=FLAGS.feature_screen_size,
        feature_minimap=FLAGS.feature_minimap_size,
        rgb_screen=FLAGS.rgb_screen_size,
        rgb_minimap=FLAGS.rgb_minimap_size,
        action_space=None,
        use_feature_units=FLAGS.use_feature_units,
        use_raw_units=FLAGS.use_raw_units),
      step_mul=FLAGS.step_mul,
      game_steps_per_episode=FLAGS.game_steps_per_episode,
      visualize=visualize) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)
    
    replay_buffer = []
    for recorder, is_done in run_loop([agent], env, MAX_AGENT_STEPS):
      if FLAGS.training:
        replay_buffer.append(recorder)
        if is_done:
          counter = 0
          with LOCK:
            global COUNTER
            COUNTER += 1
            counter = COUNTER
          # Learning rate schedule
          learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)
          agent.update(replay_buffer, FLAGS.discount, learning_rate, counter)
          replay_buffer = []
          if counter % FLAGS.snapshot_step == 1:
            agent.save_model(SNAPSHOT, counter)
          if counter >= FLAGS.max_steps:
            break
          obs = recorder[-1].observation
          score = obs["score_cumulative"][0]
          print('Your score is '+str(score)+'!')
      elif is_done:
        obs = recorder[-1].observation
        score = obs["score_cumulative"][0]
        print('Your score is '+str(score)+'!')
    if FLAGS.save_replay:
      env.save_replay(agent.name)


def main(unused_argv):
  """Run an agent."""
  if FLAGS.trace:
    stopwatch.sw.trace()
  elif FLAGS.profile:
    stopwatch.sw.enable()

  maps.get(FLAGS.map)

  players = []
  players.append(sc2_env.Agent(sc2_env.Race["random"], "A3CAgent"))

  agents = []
  for i in range(PARALLEL):
    agent = A3CAgent(FLAGS.training, int(FLAGS.feature_minimap_size[0]), int(FLAGS.feature_screen_size[0]))
    agent.build_model(i > 0, DEVICE[i % len(DEVICE)])
    agents.append(agent)

  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  summary_writer = tf.summary.FileWriter(LOG)
  for i in range(PARALLEL):
    agents[i].setup(sess, summary_writer)
  agent.initialize()

  if not FLAGS.training or FLAGS.continuation:
    global COUNTER
    COUNTER = agent.load_model(SNAPSHOT)

  threads = []
  for _ in range(PARALLEL - 1):
    t = threading.Thread(target=run_thread, args=(agents[i], players, FLAGS.map, False))
    threads.append(t)
    t.daemon = True
    t.start()

  run_thread(agents[-1], players, FLAGS.map, FLAGS.render)

  for t in threads:
    t.join()

  if FLAGS.profile:
    print(stopwatch.sw)


if __name__ == "__main__":
  app.run(main)