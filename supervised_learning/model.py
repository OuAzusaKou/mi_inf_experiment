from typing import List, Type

import numpy as np
import torch
from torch import nn

input_size = 30 + 20 + 2 + 1

hidden_size = 64

output_size = 20


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
  """
  Create a multi layer perceptron (MLP), which is
  a collection of fully-connected layers each followed by an activation function.

  :param input_dim: Dimension of the input vector
  :param output_dim:
  :param net_arch: Architecture of the neural net
      It represents the number of units per layer.
      The length of this list is the number of layers.
  :param activation_fn: The activation function
      to use after each layer.
  :param squash_output: Whether to squash the output using a Tanh
      activation function
  :return:
  """

  if len(net_arch) > 0:
    modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
  else:
    modules = []

  for idx in range(len(net_arch) - 1):
    modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
    modules.append(activation_fn())

  if output_dim > 0:
    last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
    modules.append(nn.Linear(last_layer_dim, output_dim))
  if squash_output:
    modules.append(nn.Tanh())
  return modules




class GEN_MLP(nn.Module):
  def __init__(self, size_list):
    super(GEN_MLP, self).__init__()

    self.mlplayer = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.ReLU(True),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(True),
    # 最后一层不需要添加激活函数
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(True),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(True),
    # 最后一层不需要添加激活函数
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(True),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(True),
    # 最后一层不需要添加激活函数
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(True),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(True),
    # 最后一层不需要添加激活函数
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(True),
      nn.Linear(hidden_size, output_size),

      nn.Tanh(),
    )



  def forward(self, x):
    batch_size = len(x)



    x_identical = x.clone()

    output_ = self.mlplayer(x)

    output = torch.zeros_like(output_)

    for i in range(5):

      output[:, i*4 + 0] = x_identical[:, -3] / 2 + x_identical[:, -3] / 2 * output_[:,i*4 + 0]

      output[:, i*4 + 1] = x_identical[:, -2] /2 + x_identical[:, -2] / 2 * output_[:,i*4 + 1]
      output[:, i*4 + 2] = (x_identical[:, 30 + i*4 + 0] + x_identical[:, 30 + i * 4 + 1]) / 2 + \
                           (x_identical[:, 30 + i * 4 + 1] - x_identical[:, 30 + i*4 + 0]) / 2 * output_[:, i*4 + 2]

      output[:, i*4 + 3] = (x_identical[:, 30 + i * 4 + 2] + x_identical[:, 30 + i * 4 + 3]) / 2 + \
                           (x_identical[:, 30 + i * 4 + 3] - x_identical[:, 30 + i * 4 + 2]) / 2 * output_[:, i*4 + 3]


    return output


class MIMlp(nn.Module):
  def __init__(self, size_list,
               activation_fn: Type[nn.Module] = nn.ReLU,):
    super(MIMlp, self).__init__()

    self.first_layer = nn.Sequential(nn.Linear(size_list[0], size_list[1]), activation_fn())

    self.second_layer = nn.Sequential(nn.Linear(size_list[1], size_list[2]), activation_fn())

    self.last_layer = nn.Sequential(nn.Linear(size_list[2], size_list[3]))

  def forward(self, x):

    x = x.reshape((-1, 28*28))

    x1 = self.first_layer(x)

    x2 = self.second_layer(x1)

    output = self.last_layer(x2)

    return output

