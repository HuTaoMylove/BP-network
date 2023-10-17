import torch
import copy

import nn


class Basic_optim:
    def __init__(self, net: nn.Sequential, lr=0.001, weight_decay=0.01):
        self.lr = lr
        self.net = net
        self.weight_decay = weight_decay

    def zero_grad(self):
        pass

    def step(self):
        pass


class SGD(Basic_optim):
    def __init__(self, net: nn.Sequential, lr=0.001, weight_decay=0.001):
        super(SGD, self).__init__(net, lr, weight_decay)

    def zero_grad(self):
        self.net.zero_grad()

    def step(self):
        if self.weight_decay > 0:
            for i in range(len(self.net.model_list)):
                if self.net.model_list[i].w is not None:
                    self.net.model_list[i].w.data = self.net.model_list[i].w.data * (1 - self.weight_decay)
                if self.net.model_list[i].b is not None:
                    self.net.model_list[i].b.data = self.net.model_list[i].b.data * (1 - self.weight_decay)
        self.net.step(self.lr)


