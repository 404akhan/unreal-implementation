import random
import numpy as np
from collections import deque


class ExperienceFrame(object):
  def __init__(self, state, action, reward, next_state, done):
    self.state = state
    self.action = action 
    self.reward = np.clip(reward, -1, 1) 
    self.next_state = next_state 
    self.done = done 

class Experience(object):
  def __init__(self, history_size):
    self._history_size = history_size
    self._frames = deque(maxlen=history_size)
    self._zero_reward_indices = deque(maxlen=history_size)
    self._non_zero_reward_indices = deque(maxlen=history_size)

  def add_frame(self, frame):
    self._frames.append(frame)
    if frame.reward == 0:
      self._zero_reward_indices.append(frame)
    else:
      self._non_zero_reward_indices.append(frame)

  def is_full(self):
    return len(self._frames) >= self._history_size

  def ready_rp(self):
    return len(self._zero_reward_indices) >= 10 and len(self._non_zero_reward_indices) >= 10 

  def sample_sequence(self, sequence_size):
    # -5 for the case if start pos is the terminated frame.
    # (Then +1 not to start from terminated frame.)
    start_pos = np.random.randint(0, len(self._frames) - sequence_size + 1)

    sampled_frames = []
    
    for i in range(sequence_size):
      frame = self._frames[start_pos+i]
      sampled_frames.append(frame)
      if frame.done:
        break
    
    return sampled_frames

  def sample_rp_sequence(self, sequence_size=1):
    from_zero = True
    if np.random.randint(2) == 1 and len(self._non_zero_reward_indices) > 0:
      from_zero = False
    
    if from_zero:
      start_pos = np.random.randint(0, len(self._zero_reward_indices) - sequence_size + 1)
    if not from_zero:
      start_pos = np.random.randint(0, len(self._non_zero_reward_indices) - sequence_size + 1)

    sampled_frames = []
    
    for i in range(sequence_size):
      if from_zero:
        frame = self._zero_reward_indices[start_pos+i]
      if not from_zero:
        frame = self._non_zero_reward_indices[start_pos+i]

      sampled_frames.append(frame)
      if frame.done:
        break
    
    return sampled_frames