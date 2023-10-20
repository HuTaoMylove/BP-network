import torch

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

class SGDM(SGD):
    def __init__(self, net: nn.Sequential, lr=0.001, momentum=0.9, weight_decay=0.001):
        super(SGDM, self).__init__(net, lr, weight_decay)
        self.momentum = momentum

    def step(self):
        for i in range(len(self.net.model_list)):
            if self.net.model_list[i].w is not None:
                self.net.model_list[i].w.data = self.net.model_list[i].w.data * (1 - self.weight_decay)
                self.net.model_list[i].w.grad = self.net.model_list[i].w.grad * self.momentum + self.net.model_list[
                    i].w_history_grad * (1 - self.momentum)
            if self.net.model_list[i].b is not None:
                self.net.model_list[i].b.data = self.net.model_list[i].b.data * (1 - self.weight_decay)
                self.net.model_list[i].b.grad = self.net.model_list[i].b.grad * self.momentum + self.net.model_list[
                    i].b_history_grad * (1 - self.momentum)
        self.net.step(self.lr)


