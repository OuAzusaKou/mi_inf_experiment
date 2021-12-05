import numpy as np
import torch
from torch import nn

import csv

device = torch.device("cpu")

class Degree_loss(nn.Module):
  def __init__(self):
    super().__init__()
    self.loss = nn.MSELoss()

  def forward(self, x, y):

    y=torch.div(y,180)
    #print(y.size())
    x=x.flatten()
    #print(x.size())
    a=torch.stack([(x-y).pow(2),(x-y-2).pow(2),(x-y+2).pow(2)],dim=1)
    b=torch.min(a,dim=1)[0]
    #print(b.size())
    return torch.mean(b)




  def get_reward(self,RELATION_TARGET):

      reward_ = 0

      for i in range(5):
          for j in range(5):
              #print(RELATION_TARGET[i, j])
              self.Score_mach.reset_room(room1 = i, room2 = j ,state= self.state, grid_size= self.grid_size)
              if RELATION_TARGET[i, j] == -1:
                  pass
              elif RELATION_TARGET[i, j] == 2:
                  reward_ += self.Score_mach.need_externally_tangent() * 10
              elif RELATION_TARGET[i, j] == 1:
                  reward_ += self.Score_mach.need_seperated() * 15

              reward_ += self.Score_mach.union_set()*0.1
          reward_+= self.Score_mach.need_inside_boundary(boundary= 0, room = i) * 30
      #print(reward_)
      return reward_

class Diversity_loss(nn.Module):

    def __init__(self):
        super().__init__()

        self.pwdis = torch.nn.PairwiseDistance(p=2)
    def similarity(self,answer1,answer2):

        dis = self.pwdis(answer1, answer2)

        return dis

    def forward(self, x, query, OutPut_DataSet):

        answers = OutPut_DataSet.sample_answers(query)

        similarity_buf = 0

        answers_length = len(answers)
        #print('aw',answers)
        for answer in answers:
            similarity_buf += self.similarity(torch.tensor(answer), x)

        diversity = similarity_buf / answers_length

        return diversity


delta = 5


def gausin_distance(x, y, delta=delta):
    H = torch.norm(x - y).to(device)
    distance = torch.exp(-H / 2 / (delta ** 2)).to(device)

    return distance


def gram_matrix(data, kernel=gausin_distance):
    # num=data.shape[0]
    # matrix=torch.zeros((num,num)).cuda()
    # for i in range (0,num):
    #   for j in range (0,num):
    #        matrix[i][j]=kernel(data[i],data[j])
    num = data.shape[0]
    #print(data.shape)
    datav3 = torch.mm(data, torch.transpose(data, 0, 1).to(device)).to(device)
    datav1 = torch.diag(datav3, 0).to(device)
    # print(datav1.shape)
    # matrix=torch.sqrt(datav1+datav2-2*datav3)
    buf1 = ((-2) * datav3 + datav1).to(device)
    buf2 = torch.transpose(buf1, 0, 1).to(device)
    buf3 = (buf2 + datav1).to(device)
    # print(buf3)
    matrix = torch.exp(-buf3 / 2 / (delta ** 2)).to(device)
    return matrix


def emerinal_hsic(X, Y):
    num = X.shape[0]
    # print(num)

    Kx = gram_matrix(X)

    Ky = gram_matrix(Y)

    H = torch.eye(num).to(device) - torch.ones((num, num), dtype=torch.float32).to(device) / num
    # print(H)
    hsic = 1 / (num - 1) * torch.trace(torch.mm(torch.mm(torch.mm(Kx, H).to(device), Ky).to(device), H).to(device)).to(device)

    return hsic





class hsic_loss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, x, y):
        x = x.reshape((-1, 28*28))
        mi_input_output = hsic_normalized(x, y,sigma=None, use_cuda=False, to_numpy=False)

        mi_component_sum = 0

        for i in range((y.shape[-1])):
            for j in range(i+1,(y.shape[-1])):
                mi_component =  hsic_normalized(y[:,i].reshape((-1, 1)), y[:,j].reshape((-1, 1)), sigma=None, use_cuda=False, to_numpy=False)
                mi_component_sum += mi_component

        loss =  -mi_input_output + mi_component_sum

        return loss




def sigma_estimation(X, Y):
    """ sigma from median distance
    """
    D = distmat(torch.cat([X,Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med=np.mean(Tri)
    if med<1E-2:
        med=1E-2
    return med

def distmat(X):
    """ distance matrix
    """
    r = torch.sum(X*X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X,0,1))
    D = r.expand_as(a) - 2*a +  torch.transpose(r,0,1).expand_as(a)
    return D

def kernelmat(X, sigma):
    """ kernel matrix baker
    """
    m = int(X.size()[0])
    H = torch.eye(m) - (1./m) * torch.ones([m,m])
    Dxx = distmat(X)
    if sigma:
        Kx = torch.exp( -Dxx / (2.*sigma*sigma)).type(torch.FloatTensor)   # kernel matrices
    else:
        try:
            sx = sigma_estimation(X,X)
            Kx = torch.exp( -Dxx / (2.*sx*sx)).type(torch.FloatTensor)
        except RuntimeError as e:
            raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                sx, torch.max(X), torch.min(X)))
    Kxc = torch.mm(Kx,H)
    return Kxc

def distcorr(X, sigma=1.0):
    X = distmat(X)
    X = torch.exp( -X / (2.*sigma*sigma))
    return torch.mean(X)

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def mmd(x, y, sigma=None, use_cuda=True, to_numpy=False):
    m = int(x.size()[0])
    H = torch.eye(m) - (1./m) * torch.ones([m,m])
    # H = Variable(H)
    Dxx = distmat(x)
    Dyy = distmat(y)

    if sigma:
        Kx  = torch.exp( -Dxx / (2.*sigma*sigma))   # kernel matrices
        Ky  = torch.exp( -Dyy / (2.*sigma*sigma))
        sxy = sigma
    else:
        sx = sigma_estimation(x,x)
        sy = sigma_estimation(y,y)
        sxy = sigma_estimation(x,y)
        Kx = torch.exp( -Dxx / (2.*sx*sx))
        Ky = torch.exp( -Dyy / (2.*sy*sy))
    # Kxc = torch.mm(Kx,H)            # centered kernel matrices
    # Kyc = torch.mm(Ky,H)
    Dxy = distmat(torch.cat([x,y]))
    Dxy = Dxy[:x.size()[0], x.size()[0]:]
    Kxy = torch.exp( -Dxy / (1.*sxy*sxy))

    mmdval = torch.mean(Kx) + torch.mean(Ky) - 2*torch.mean(Kxy)

    return mmdval

def mmd_pxpy_pxy(x,y,sigma=None,use_cuda=True, to_numpy=False):
    """
    """
    if use_cuda:
        x = x.cuda()
        y = y.cuda()
    m = int(x.size()[0])

    Dxx = distmat(x)
    Dyy = distmat(y)
    if sigma:
        Kx  = torch.exp( -Dxx / (2.*sigma*sigma))   # kernel matrices
        Ky  = torch.exp( -Dyy / (2.*sigma*sigma))
    else:
        sx = sigma_estimation(x,x)
        sy = sigma_estimation(y,y)
        sxy = sigma_estimation(x,y)
        Kx = torch.exp( -Dxx / (2.*sx*sx))
        Ky = torch.exp( -Dyy / (2.*sy*sy))
    A = torch.mean(Kx*Ky)
    B = torch.mean(torch.mean(Kx,dim=0)*torch.mean(Ky, dim=0))
    C = torch.mean(Kx)*torch.mean(Ky)
    mmd_pxpy_pxy_val = A - 2*B + C
    return mmd_pxpy_pxy_val

def hsic_regular(x, y, sigma=None, use_cuda=True, to_numpy=False):
    """
    """
    Kxc = kernelmat(x, sigma)
    Kyc = kernelmat(y, sigma)
    KtK = torch.mul(Kxc, Kyc.t())
    Pxy = torch.mean(KtK)
    return Pxy

def hsic_normalized(x,y,sigma=None, use_cuda=True, to_numpy=True):
    """
    """
    m = int(x.size()[0])
    Pxy = hsic_regular(x, y, sigma, use_cuda)
    Px = torch.sqrt(hsic_regular(x, x, sigma, use_cuda))
    Py = torch.sqrt(hsic_regular(y, y, sigma, use_cuda))
    thehsic = Pxy/(Px*Py)
    return thehsic

def hsic_normalized_cca(x,y,sigma=None, use_cuda=True, to_numpy=True):
    """
    """
    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma)
    Kyc = kernelmat(y, sigma)
    epsilon = 1E-5
    K_I = torch.eye(m)
    Kxc_i = torch.inverse(Kxc + epsilon*m*K_I)
    Kyc_i = torch.inverse(Kyc + epsilon*m*K_I)
    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))
    return Pxy
