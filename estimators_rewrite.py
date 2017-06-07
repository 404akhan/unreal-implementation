import numpy as np
import tensorflow as tf

from constants import *

def build_shared_network(X):
  
  conv1 = tf.contrib.layers.conv2d(
    X, 16, 8, 4, activation_fn=tf.nn.relu, scope="conv1")
  conv2 = tf.contrib.layers.conv2d(
    conv1, 32, 4, 2, activation_fn=tf.nn.relu, scope="conv2")

  return conv2

class Model():
	def __init__(self, num_outputs):
		self.num_outputs = num_outputs

		self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name='X')
		self.targets_pi = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
		self.targets_v = tf.placeholder(shape=[None], dtype=tf.float32, name="y")	
		self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

		X = tf.to_float(self.states) / 255.0
		batch_size = tf.shape(self.states)[0]

		with tf.variable_scope("shared", reuse=False):
			conv2 = build_shared_network(X)
			
		# Fully connected layer
		fc1 = tf.contrib.layers.fully_connected(
			inputs=tf.contrib.layers.flatten(conv2),
			num_outputs=256,
			scope="fc1")

		### Policy
		self.logits_pi = tf.contrib.layers.fully_connected(fc1, num_outputs, activation_fn=None)
		self.probs_pi = tf.nn.softmax(self.logits_pi) + 1e-8

		self.predictions_pi = {
			"logits": self.logits_pi,
			"probs": self.probs_pi
		}

		# We add entropy to the loss to encourage exploration
		self.entropy_pi = -tf.reduce_sum(self.probs_pi * tf.log(self.probs_pi), 1, name="entropy")
		self.entropy_mean_pi = tf.reduce_mean(self.entropy_pi, name="entropy_mean")

		# Get the predictions for the chosen actions only
		gather_indices_pi = tf.range(batch_size) * tf.shape(self.probs_pi)[1] + self.actions
		self.picked_action_probs_pi = tf.gather(tf.reshape(self.probs_pi, [-1]), gather_indices_pi)

		self.losses_pi = - (tf.log(self.picked_action_probs_pi) * self.targets_pi + 0.01 * self.entropy_pi)
		self.loss_pi = tf.reduce_sum(self.losses_pi, name="loss_pi")

		### Value
		self.logits_v = tf.contrib.layers.fully_connected(
			inputs=fc1,
			num_outputs=1,
			activation_fn=None,
			scope='logits_value')
		self.logits_v = tf.squeeze(self.logits_v, squeeze_dims=[1])

		self.predictions_v = {
			"logits": self.logits_v
		}

		self.losses_v = tf.squared_difference(self.logits_v, self.targets_v)
		self.loss_v = tf.reduce_sum(self.losses_v, name="loss_v")
		
		# Combine loss
		self.loss = self.loss_pi + self.loss_v
		
		if USE_REWARD_PREDICTION:
			### Reward prediction
			self._build_rp_network()
			self.loss = self.loss + self.rp_loss
		
		if USE_VALUE_REPLAY:
			### Value Replay
			self._build_vr_network()
			self.loss = self.loss + self.vr_loss

		# Train
		self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
		self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
		self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
		self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
			global_step=tf.contrib.framework.get_global_step())

	def _build_rp_network(self):
		self.rp_states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32)
		self.rp_lambda = tf.placeholder(dtype=tf.float32, shape=())

		X = tf.to_float(self.rp_states) / 255.0

		with tf.variable_scope("shared", reuse=True):
			conv2 = build_shared_network(X)

		fc1 = tf.contrib.layers.fully_connected(
			inputs=tf.contrib.layers.flatten(conv2),
			num_outputs=128,
			scope="rp_fc1")

		self.rp_logits = tf.contrib.layers.fully_connected(fc1, 3, activation_fn=None)
		self.rp_c = tf.nn.softmax(self.rp_logits) + 1e-8

		self.rp_c_targets = tf.placeholder(shape=[None, 3], dtype=tf.float32)

		self.rp_loss = -tf.reduce_sum(self.rp_c_targets * tf.log(self.rp_c))
		self.rp_loss = self.rp_lambda * self.rp_loss

	def _build_vr_network(self):
		self.vr_states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32)
		self.vr_value_targets = tf.placeholder(shape=[None], dtype=tf.float32)
		self.vr_lambda = tf.placeholder(dtype=tf.float32, shape=())

		X = tf.to_float(self.vr_states) / 255.0

		with tf.variable_scope("shared", reuse=True):
			conv2 = build_shared_network(X)

		fc1 = tf.contrib.layers.fully_connected(
			inputs=tf.contrib.layers.flatten(conv2),
			num_outputs=256,
			scope="fc1",
			reuse=True)

		self.vr_value = tf.contrib.layers.fully_connected(
			inputs=fc1,
			num_outputs=1,
			activation_fn=None,
			scope='logits_value',
			reuse=True)

		self.vr_value = tf.squeeze(self.vr_value, squeeze_dims=[1])

		self.vr_losses = tf.squared_difference(self.vr_value, self.vr_value_targets)
		self.vr_loss = tf.reduce_sum(self.vr_losses)
		self.vr_loss = self.vr_lambda * self.vr_loss
