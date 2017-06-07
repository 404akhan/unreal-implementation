import gym
import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)

# from lib import plotting
from lib.atari.state_processor import StateProcessor
from lib.atari import helpers as atari_helpers
from estimators_rewrite import Model
from experience import Experience, ExperienceFrame
from constants import *

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


def make_copy_params_op(v1_list, v2_list):
  """
  Creates an operation that copies parameters from variable in v1_list to variables in v2_list.
  The ordering of the variables in the lists must be identical.
  """
  v1_list = list(sorted(v1_list, key=lambda v: v.name))
  v2_list = list(sorted(v2_list, key=lambda v: v.name))

  update_ops = []
  for v1, v2 in zip(v1_list, v2_list):
    op = v2.assign(v1)
    update_ops.append(op)

  return update_ops

def make_train_op(local_estimator, global_estimator):
  """
  Creates an op that applies local estimator gradients
  to the global estimator.
  """
  local_grads, _ = zip(*local_estimator.grads_and_vars)
  # Clip gradients
  local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)
  _, global_vars = zip(*global_estimator.grads_and_vars)
  local_global_grads_and_vars = list(zip(local_grads, global_vars))
  return global_estimator.optimizer.apply_gradients(local_global_grads_and_vars,
          global_step=tf.contrib.framework.get_global_step())


class Worker(object):
  """
  An A3C worker thread. Runs episodes locally and updates global shared value and policy nets.

  Args:
    name: A unique name for this worker
    env: The Gym environment used by this worker
    policy_net: Instance of the globally shared policy net
    value_net: Instance of the globally shared value net
    global_counter: Iterator that holds the global step
    discount_factor: Reward discount factor
    summary_writer: A tf.train.SummaryWriter for Tensorboard summaries
    max_global_steps: If set, stop coordinator when global_counter > max_global_steps
  """
  def __init__(self, name, env, model_net, global_counter, discount_factor=0.99, summary_writer=None, max_global_steps=None):
    self.name = name
    self.discount_factor = discount_factor
    self.max_global_steps = max_global_steps
    self.global_step = tf.contrib.framework.get_global_step()
    self.global_model_net = model_net
    self.global_counter = global_counter
    self.local_counter = itertools.count()
    self.sp = StateProcessor()
    self.summary_writer = summary_writer
    self.env = env

    # Create local policy/value nets that are not updated asynchronously
    with tf.variable_scope(name):
      self.model_net = Model(model_net.num_outputs)

    # Op to copy params from global policy/valuenets
    self.copy_params_op = make_copy_params_op(
      tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
      tf.contrib.slim.get_variables(scope=self.name, collection=tf.GraphKeys.TRAINABLE_VARIABLES))

    self.mnet_train_op = make_train_op(self.model_net, self.global_model_net)

    self.experience = Experience(EXP_HIST_SIZE)
    self.state = None

  def run(self, sess, coord, t_max):
    with sess.as_default(), sess.graph.as_default():
      # Initial state
      self.state = atari_helpers.atari_make_initial_state(self.sp.process(self.env.reset()))
      try:
        while not coord.should_stop():
          # Copy Parameters from the global networks
          sess.run(self.copy_params_op)

          # Collect some experience
          transitions, local_t, global_t = self.run_n_steps(t_max, sess)

          if self.max_global_steps is not None and global_t >= self.max_global_steps:
            tf.logging.info("Reached global step {}. Stopping.".format(global_t))
            coord.request_stop()
            return

          # Update the global networks
          self.update(transitions, sess)

      except tf.errors.CancelledError:
        return

  def _policy_net_predict(self, state, sess):
    feed_dict = { self.model_net.states: [state] }
    preds = sess.run(self.model_net.predictions_pi, feed_dict)
    return preds["probs"][0]

  def _value_net_predict(self, state, sess):
    feed_dict = { self.model_net.states: [state] }
    preds = sess.run(self.model_net.predictions_v, feed_dict)
    return preds["logits"][0]

  def _run_vr_value(self, state, sess):
    vr_v_out = sess.run( self.model_net.vr_value,
                         feed_dict = {self.model_net.vr_states : [state]} )
    return vr_v_out[0]

  def run_n_steps(self, n, sess):
    transitions = []
    for _ in range(n):
      # Take a step
      action_probs = self._policy_net_predict(self.state, sess)
      action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
      next_state, reward, done, _ = self.env.step(action)
      next_state = atari_helpers.atari_make_next_state(self.state, self.sp.process(next_state))

      frame = ExperienceFrame(
        state=self.state, action=action, reward=reward, next_state=next_state, done=done)
      self.experience.add_frame(frame)

      # Store transition
      transitions.append(Transition(
        state=self.state, action=action, reward=reward, next_state=next_state, done=done))

      # Increase local and global counters
      local_t = next(self.local_counter)
      global_t = next(self.global_counter)

      if local_t % 100 == 0:
        tf.logging.info("{}: local Step {}, global step {}".format(self.name, local_t, global_t))

      if done:
        self.state = atari_helpers.atari_make_initial_state(self.sp.process(self.env.reset()))
        break
      else:
        self.state = next_state
    return transitions, local_t, global_t

  def _process_rp(self, sess):
    # [Reward prediction]
    # default sample size 1
    transitions = self.experience.sample_rp_sequence()

    states = []
    reward_classes = []

    for transition in transitions[::-1]:
      reward_class = np.zeros(3)

      if transition.reward == 0:
        reward_class[0] = 1.0 # zero
      elif transition.reward > 0:
        reward_class[1] = 1.0 # positive
      else:
        reward_class[2] = 1.0 # negative
      
      states.append(transition.state)
      reward_classes.append(reward_class)

    return states, reward_classes

  def _process_vr(self, sess):
    # [Value replay]
    # Sample 20+1 frame (+1 for last next state)
    transitions = self.experience.sample_sequence(T_MAX)

    reward = 0.0
    if not transitions[-1].done:
      reward = self._run_vr_value(transitions[-1].next_state, sess)

    # Accumulate minibatch exmaples
    states = []
    value_targets = []

    for transition in transitions[::-1]:
      reward = transition.reward + self.discount_factor * reward
      # Accumulate updates
      states.append(transition.state)
      value_targets.append(reward)

    return states, value_targets

  def update(self, transitions, sess):
    """
    Updates global policy and value networks based on collected experience

    Args:
      transitions: A list of experience transitions
      sess: A Tensorflow session
    """

    # If we episode was not done we bootstrap the value from the last state
    reward = 0.0
    if not transitions[-1].done:
      reward = self._value_net_predict(transitions[-1].next_state, sess)

    # Accumulate minibatch exmaples
    states = []
    policy_targets = []
    value_targets = []
    actions = []

    for transition in transitions[::-1]:
      reward = transition.reward + self.discount_factor * reward
      policy_target = (reward - self._value_net_predict(transition.state, sess))
      # Accumulate updates
      states.append(transition.state)
      actions.append(transition.action)
      policy_targets.append(policy_target)
      value_targets.append(reward)

    feed_dict = {
      self.model_net.states: np.array(states),
      self.model_net.targets_pi: policy_targets,
      self.model_net.targets_v: value_targets,
      self.model_net.actions: actions,
    }

    if USE_REWARD_PREDICTION:
      rp_lambda = 0
      if self.experience.ready_rp(): rp_lambda = 1

      rp_states, rp_c_targets = self._process_rp(sess)
      rp_feed_dict = {
        self.model_net.rp_states: rp_states,
        self.model_net.rp_c_targets: rp_c_targets,
        self.model_net.rp_lambda: rp_lambda
      }
      feed_dict.update(rp_feed_dict)

    if USE_VALUE_REPLAY:
      vr_lambda = 0
      if self.experience.is_full(): vr_lambda = 1
      
      vr_states, vr_value_targets = self._process_vr(sess)
      vr_feed_dict = {
        self.model_net.vr_states: vr_states,
        self.model_net.vr_value_targets: vr_value_targets,
        self.model_net.vr_lambda: vr_lambda
      }
      feed_dict.update(vr_feed_dict)

    # Train the global estimators using local gradients
    global_step, mnet_loss, _ = sess.run([
      self.global_step,
      self.model_net.loss,
      self.mnet_train_op
    ], feed_dict)

    return mnet_loss