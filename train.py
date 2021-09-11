import torch
import torch.nn as nn
import torch.nn.functional as F

class Trainer_MR():
    def __init__(self, model:nn.Module, device):
        self.model = model
        self.torch_device = device
    
    def loss_func(out, y):
        return F.binary_cross_entropy_with_logits(out, y)
    
    def train(self, inputs, input_test, label, label_test, epoch:int):
        seqs, others = inputs
        seqs_test, others_test = input_test
        self.model.to(self.torch_device)
        self.model.train()
        optm = torch.optim.Adam(self.model.parameters())
        loss_list = []
        test_loss_list = []
        for epoch in range(epoch):
            optm.zero_grad()
            out = self.model.forward(seqs, others)
            loss = self.loss_func(out, label)
            test_loss = self.loss_func(self.model(seqs_test, others_test), label_test).item()
            loss.backward()
            optm.step()
            sep_loss = loss.item()
            loss_list.append(sep_loss)
            test_loss_list.append(test_loss)
            print(epoch, sep_loss, test_loss)
        return loss_list, test_loss_list