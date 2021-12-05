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
from supervised_learning.test_loop import test_loop, mnist_test_loop
from supervised_learning.train_loop import mi_train_loop, mnist_train_loop
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
  model_feature = MIMlp(size_list=[28*28, 200, 30, 10]).to(device)
  model_classifier = nn.Linear(10, 10)
  model = nn.Sequential(model_feature, model_classifier)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model_classifier.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

  trained_Flag = True

  training_data = datasets.MNIST('./mnist',  train=True,
                                 transform=torchvision.transforms.ToTensor(), download=True)
  test_data = datasets.MNIST('./mnist', train=False,
                                 transform=torchvision.transforms.ToTensor(), download=True)
  train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=8)
  test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8)


  epochs = 50000

  writer = SummaryWriter('runs/scalar_example')
  model_classifier.apply(weight_init)

  if trained_Flag == True:
    model_feature.load_state_dict(torch.load('snn_model_weights.pth'))
  for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")

    mnist_train_loop(train_dataloader, model, loss_fn, optimizer, writer, t, device)

    if t % 100 == 0:
      torch.save(model.state_dict(), 'finetuning_mi.pth')
      #model.load_state_dict(torch.load('snn_model_weights.pth'))
      model.eval()
      mnist_test_loop(test_dataloader, model, loss_fn, writer, t, device)
  print("Done!")

