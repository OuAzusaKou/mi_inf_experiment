import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from tensorboardX import SummaryWriter
from torchvision import datasets,transforms

from supervised_learning.loss import hsic_loss
from supervised_learning.model import GEN_MLP, MIMlp
from supervised_learning.test_loop import test_loop
from supervised_learning.train_loop import mi_train_loop
from supervised_learning.weightinit import weight_init

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


if __name__ == '__main__':
  torch.cuda.empty_cache()

  batch_size = 128
  capacity = 50







  torch.set_default_tensor_type(torch.FloatTensor)
  learning_rate = 1e-5
  #model = NeuralNetwork()
  model = MIMlp(size_list=[28*28, 200, 30, 10]).to(device)
  loss_fn = hsic_loss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

  trained_Flag = False

  training_data = datasets.MNIST('./mnist',  train=True,
                                 transform=torchvision.transforms.ToTensor(), download=True)
  test_data = datasets.MNIST('./mnist', train=False,
                                 transform=torchvision.transforms.ToTensor(), download=True)
  train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=8)
  test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8)


  epochs = 50000

  writer = SummaryWriter('runs/scalar_example')
  model.apply(weight_init)

  if trained_Flag == True:
    model.load_state_dict(torch.load('snn_model_weights.pth'))
  for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")

    mi_train_loop(train_dataloader, model, loss_fn, optimizer, writer, t, device)

    if t % 100 == 0:
      torch.save(model.state_dict(), 'snn_model_weights.pth')
      model.load_state_dict(torch.load('snn_model_weights.pth'))
      # model.eval()
      # test_loop(test_dataloader, model, loss_fn, writer, t, device)
  print("Done!")

