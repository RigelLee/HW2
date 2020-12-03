from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

def build_net(minimap, screen, info, num_action, network_type):
  if network_type == 'Atari':
    return Atari(minimap, screen, info, num_action)
  elif network_type == 'FullyConv':
    return FullyConv(minimap, screen, info, num_action)
  elif network_type == 'FullyConvLSTM':
    return FullyConvLSTM(minimap, screen, info, num_action)
  else:
    raise 'FLAGS.net must be Atari, FullyConv or FullyConvLSTM'

def FullyConv(minimap, screen, info, num_action):
  # Extract features
  mconv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=5,
                         stride=1,
                         padding='same',
                         activation_fn=tf.nn.relu,
                         scope='mconv1')
  mconv2 = layers.conv2d(mconv1,
                         num_outputs=32,
                         kernel_size=3,
                         stride=1,
                         padding='same',
                         activation_fn=tf.nn.relu,
                         scope='mconv2')
  sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=5,
                         stride=1,
                         padding='same',
                         activation_fn=tf.nn.relu,
                         scope='sconv1')
  sconv2 = layers.conv2d(sconv1,
                         num_outputs=32,
                         kernel_size=3,
                         stride=1,
                         padding='same',
                         activation_fn=tf.nn.relu,
                         scope='sconv2')
  info_fc = layers.fully_connected(layers.flatten(info),
                                   num_outputs=256,
                                   activation_fn=tf.tanh,
                                   scope='info_fc')
  info_fc_spatial = layers.fully_connected(layers.flatten(info),
                                           num_outputs=32,
                                           activation_fn=tf.tanh,
                                           scope='info_fc_spatial')
  info_fc_spatial = tf.expand_dims(info_fc_spatial, 1)
  info_fc_spatial = tf.expand_dims(info_fc_spatial, 1)

  #expend to 2D
  for _ in range(6):
    info_fc_spatial = tf.concat([info_fc_spatial, info_fc_spatial], 1)
    info_fc_spatial = tf.concat([info_fc_spatial, info_fc_spatial], 2)

  feat_conv = tf.concat([mconv2, sconv2, info_fc_spatial], axis=3)
  spatial_action = layers.conv2d(feat_conv,
                                 num_outputs=1,
                                 kernel_size=1,
                                 stride=1,
                                 padding='same',
                                 activation_fn=tf.nn.relu,
                                 scope='spatial_action')
  spatial_action = tf.nn.softmax(layers.flatten(spatial_action))

  feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
  feat_fc = layers.fully_connected(feat_fc,
                                   num_outputs=256,
                                   activation_fn=tf.nn.relu,
                                   scope='feat_fc')
  non_spatial_action = layers.fully_connected(feat_fc,
                                              num_outputs=num_action,
                                              activation_fn=tf.nn.softmax,
                                              scope='non_spatial_action')
  value = tf.reshape(layers.fully_connected(feat_fc,
                                            num_outputs=1,
                                            activation_fn=None,
                                            scope='value'), [-1])

  return spatial_action, non_spatial_action, value

def FullyConvLSTM(minimap, screen, info, num_action):
  mconv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=5,
                         stride=1,
                         padding='same',
                         activation_fn=tf.nn.relu,
                         scope='mconv1')
  mconv2 = layers.conv2d(mconv1,
                         num_outputs=32,
                         kernel_size=3,
                         stride=1,
                         padding='same',
                         activation_fn=tf.nn.relu,
                         scope='mconv2')
  sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=5,
                         stride=1,
                         padding='same',
                         activation_fn=tf.nn.relu,
                         scope='sconv1')
  sconv2 = layers.conv2d(sconv1,
                         num_outputs=32,
                         kernel_size=3,
                         stride=1,
                         padding='same',
                         activation_fn=tf.nn.relu,
                         scope='sconv2')
  info_fc = layers.fully_connected(layers.flatten(info),
                                   num_outputs=256,
                                   activation_fn=tf.tanh,
                                   scope='info_fc')
  info_fc_spatial = layers.fully_connected(layers.flatten(info),
                                           num_outputs=32,
                                           activation_fn=tf.tanh,
                                           scope='info_fc_spatial')
  info_fc_spatial = tf.expand_dims(info_fc_spatial, 1)
  info_fc_spatial = tf.expand_dims(info_fc_spatial, 1)
  for _ in range(6):
    info_fc_spatial = tf.concat([info_fc_spatial, info_fc_spatial], 1)
    info_fc_spatial = tf.concat([info_fc_spatial, info_fc_spatial], 2)
  feat_conv = tf.concat([mconv2, sconv2, info_fc_spatial], axis=3)
  spatial_action = layers.conv2d(feat_conv,
                                 num_outputs=1,
                                 kernel_size=1,
                                 stride=1,
                                 padding='same',
                                 activation_fn=tf.nn.relu,
                                 scope='spatial_action')
  spatial_action = tf.nn.softmax(layers.flatten(spatial_action))

  # Compute non spatial actions and value
  feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
  feat_fc = layers.fully_connected(feat_fc,
                                   num_outputs=256,
                                   activation_fn=tf.nn.relu,
                                   #weights_regularizer=l2_regularizer,
                                   scope='feat_fc')
  cell = tf.nn.rnn_cell.BasicLSTMCell(256)
  feat_fc1 = tf.expand_dims(feat_fc, 0)
  istate = cell.zero_state(1, dtype = tf.float32)
  output_rnn, _ = tf.nn.dynamic_rnn(cell, feat_fc1, initial_state = istate)
  output_rnn = tf.reduce_sum(output_rnn, 0)
  non_spatial_action = layers.fully_connected(output_rnn,
                                              num_outputs=num_action,
                                              activation_fn=tf.nn.softmax,
                                              scope='non_spatial_action')
  value = tf.reshape(layers.fully_connected(output_rnn,
                                            num_outputs=1,
                                            activation_fn=None,
                                            scope='value'), [-1])

  return spatial_action, non_spatial_action, value

def Atari(minimap, screen, info, num_action):
  # Extract features
  ssize = int(screen.shape[2])

  mconv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=8,
                         stride=4,
                         activation_fn=tf.nn.relu,
                         scope='mconv1')
  mconv2 = layers.conv2d(mconv1,
                         num_outputs=32,
                         kernel_size=4,
                         stride=2,
                         activation_fn=tf.nn.relu,
                         scope='mconv2')
  sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=8,
                         stride=4,
                         activation_fn=tf.nn.relu,
                         scope='sconv1')
  sconv2 = layers.conv2d(sconv1,
                         num_outputs=32,
                         kernel_size=4,
                         stride=2,
                         activation_fn=tf.nn.relu,
                         scope='sconv2')
  info_fc = layers.fully_connected(layers.flatten(info),
                                   num_outputs=256,
                                   activation_fn=tf.tanh,
                                   scope='info_fc')

  # Compute spatial actions, non spatial actions and value
  feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
  feat_fc = layers.fully_connected(feat_fc,
                                   num_outputs=256,
                                   activation_fn=tf.nn.relu,
                                   scope='feat_fc')

  spatial_action_x = layers.fully_connected(feat_fc,
                                            num_outputs=ssize,
                                            activation_fn=tf.nn.softmax,
                                            scope='spatial_action_x')
  spatial_action_y = layers.fully_connected(feat_fc,
                                            num_outputs=ssize,
                                            activation_fn=tf.nn.softmax,
                                            scope='spatial_action_y')
  spatial_action_x = tf.reshape(spatial_action_x, [-1, 1, ssize])
  spatial_action_x = tf.tile(spatial_action_x, [1, ssize, 1])
  spatial_action_y = tf.reshape(spatial_action_y, [-1, ssize, 1])
  spatial_action_y = tf.tile(spatial_action_y, [1, 1, ssize])
  spatial_action = layers.flatten(spatial_action_x * spatial_action_y)

  non_spatial_action = layers.fully_connected(feat_fc,
                                              num_outputs=num_action,
                                              activation_fn=tf.nn.softmax,
                                              scope='non_spatial_action')
  value = tf.reshape(layers.fully_connected(feat_fc,
                                            num_outputs=1,
                                            activation_fn=None,
                                            scope='value'), [-1])

  return spatial_action, non_spatial_action, value