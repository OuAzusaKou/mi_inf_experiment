import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomSoundDataset(Dataset):
  def __init__(self, labels_dir, data_dir, transform=None, target_transform=None):
    self.labels_dir = pd.read_csv(labels_dir)

    self.data_dir = data_dir
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.labels_dir)

  def __getitem__(self, idx):
    data1_path = os.path.join(self.data_dir, self.labels_dir.iloc[idx, 0])
    data2_path = os.path.join(self.data_dir, self.labels_dir.iloc[idx, 1])
    #print(sound_dir)
    #print(self.labels_dir.iloc[idx,:])
    #print(sound_path)
    data1 = pd.read_csv(data1_path,index_col=0)
    data2 = pd.read_csv(data2_path)
    label = pd.read_csv(data1_path,index_col=0)
    #label = self.labels_dir.iloc[idx, 2]
    if self.transform:
      for trans in self.transform:
        data = trans(data1,data2)
    if self.target_transform:
      for trans in self.target_transform:
        label = trans(label)
    # if label == 1:
    #   label = torch.tensor([0,1])
    # else :
    #   label = torch.tensor([1,0])
    sample = {"data": data, "label": label}
    #print(sample['sound'].size)
    return sample



class OutPut_DataSet(Dataset):
  """Buffer to store environment transitions."""

  def __init__(self, capacity, batch_size, query_shape, answer_shape, transform=None):
    self.capacity = capacity
    self.batch_size = batch_size
    self.transform = transform
    # the proprioceptive obs is stored as float32, pixels obs as uint8

    self.query = np.empty((capacity, *query_shape), dtype=np.float32)
    self.answer = np.empty((capacity, *answer_shape), dtype=np.float32)

    self.idx = 0
    self.full = False

  def add(self, query, answer):

    np.copyto(self.query[self.idx], query)
    np.copyto(self.answer[self.idx], answer.detach().numpy())

    self.idx = (self.idx + 1) % self.capacity
    self.full = self.full or self.idx == 0

  def sample_answers(self,query_idx):

    answers = []

    #print('que', query_idx)
    #print('qe', self.query[0])
    for i in range(self.capacity if self.full else self.idx):
      #print(i)
      if (self.query[i] == query_idx.detach().numpy()).all():
        #print('equal')
        answers.append(self.answer[i])

    return answers


    # # idxs = np.random.randint(
    # #   0, self.capacity if self.full else self.idx, size=self.batch_size
    # # )
    # #
    # # obses = self.obses[idxs]
    # # next_obses = self.next_obses[idxs]
    # #
    # # obses = torch.as_tensor(obses, device=self.device).float()
    # # actions = torch.as_tensor(self.actions[idxs], device=self.device)
    # # rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
    # # next_obses = torch.as_tensor(
    # #   next_obses, device=self.device
    # # ).float()
    # # not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
    # return obses, actions, rewards, next_obses, not_dones


  # def save(self, save_dir):
  #   if self.idx == self.last_save:
  #     return
  #   path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
  #   payload = [
  #     self.obses[self.last_save:self.idx],
  #     self.next_obses[self.last_save:self.idx],
  #     self.actions[self.last_save:self.idx],
  #     self.rewards[self.last_save:self.idx],
  #     self.not_dones[self.last_save:self.idx]
  #   ]
  #   self.last_save = self.idx
  #   torch.save(payload, path)
  #
  # def load(self, save_dir):
  #   chunks = os.listdir(save_dir)
  #   chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
  #   for chunk in chucks:
  #     start, end = [int(x) for x in chunk.split('.')[0].split('_')]
  #     path = os.path.join(save_dir, chunk)
  #     payload = torch.load(path)
  #     assert self.idx == start
  #     self.obses[start:end] = payload[0]
  #     self.next_obses[start:end] = payload[1]
  #     self.actions[start:end] = payload[2]
  #     self.rewards[start:end] = payload[3]
  #     self.not_dones[start:end] = payload[4]
  #     self.idx = end

  def __getitem__(self, idx):
    idx = np.random.randint(
      0, self.capacity if self.full else self.idx, size=1
    )
    querys = self.query[idx]
    answers = self.answer[idx]

    if self.transform:
      querys = self.transform(querys)
      answers = self.transform(answers)

    return querys,answers

  def __len__(self):
    return self.capacity