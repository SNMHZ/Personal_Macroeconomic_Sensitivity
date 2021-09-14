import torch
import torch.nn as nn
import torch.nn.functional as F

class Trainer_MR():
    def __init__(self, model:nn.Module, device):
        self.model = model
        self.torch_device = device
    
    def loss_func(self, out, y):
        return F.binary_cross_entropy_with_logits(out, y)
    
    def train(self, train_inputs, valid_inputs, epoch:int):
        self.model.to(self.torch_device)
        self.model.train()
        optm = torch.optim.Adam(self.model.parameters())
        loss_list = []
        valid_loss_list = []
        for ep in range(epoch):
          print(ep)
          sep_loss = 0
          k=0
          for train_x, train_y in train_inputs:
            print('train batch', k, end=' ')
            out = self.model.forward(train_x)
            loss = self.loss_func(out, train_y)
            optm.zero_grad()
            loss.backward()
            optm.step()
            sep_loss += loss.item()
            k+=1
          loss_list.append(sep_loss/k)
          print()
          valid_loss = 0
          k=0
          for valid_x, valid_y in valid_inputs:
            print('valid batch', k, end=' ')
            valid_loss += self.loss_func(self.model(valid_x), valid_y).item()
            k+=1
          valid_loss_list.append(valid_loss/k)
          print()
          print(ep, loss_list[-1], valid_loss_list[-1])
        return loss_list, valid_loss_list