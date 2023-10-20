import torch
import copy


class Basic:
    def __init__(self):
        self.w = None
        self.b = None
        self.forward_value = None
        self.w_history_grad = None
        self.b_history_grad = None

    def forward(self, input):
        pass

    def backward(self, input):
        pass

    def zero_grad(self):
        pass

    def step(self, lr=0.001):
        pass


class Sigmoid(Basic):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input):
        out = 1 / (1 + torch.exp(-input))
        self.forward_value = copy.deepcopy(out.detach().clone())
        return out

    def backward(self, input):
        return input * self.forward_value.transpose(1, 2) * (1 - self.forward_value.transpose(1, 2))


class Tanh(Basic):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, input):
        out = (1 - torch.exp(-2 * input)) / (1 + torch.exp(-2 * input))
        self.forward_value = copy.deepcopy(out.detach().clone())
        return out

    def backward(self, input):
        return input * (1 - self.forward_value.transpose(1, 2) * self.forward_value.transpose(1, 2))


class Linear(Basic):
    def __init__(self, input, output, bias=True):
        super(Linear, self).__init__()
        assert type(input) is int
        assert type(output) is int

        self.w = torch.rand([output, input], dtype=torch.float32, requires_grad=True)
        self.w_history_grad = torch.zeros_like(self.w)
        if bias:
            self.b = torch.rand([output, 1], dtype=torch.float32, requires_grad=True)
            self.b_history_grad = torch.zeros_like(self.b)

    def zero_grad(self):
        self.w.grad = torch.zeros_like(self.w)
        if self.b is not None:
            self.b.grad = torch.zeros_like(self.b)

    def forward(self, input):
        self.forward_value = copy.deepcopy(input.detach().clone())
        self.batchsize, self.input_channel = input.shape[0], input.shape[1]
        w = self.w.repeat(self.batchsize, 1, 1)
        if self.b is not None:
            b = self.b.repeat(self.batchsize, 1, 1)

        out = torch.bmm(w, input)
        if self.b is not None:
            return out + b
        else:
            return out

    def backward(self, input=None):
        self.w.grad = torch.bmm(input.transpose(1, 2), self.forward_value.transpose(1, 2)).mean(0) + self.w.grad
        if self.b is not None:
            self.b.grad = input.mean(0).T

        return torch.bmm(input, self.w.repeat(self.batchsize, 1, 1))

    def step(self, lr=0.001):
        self.w.data -= lr * self.w.grad
        self.w_history_grad = self.w.grad
        if self.b is not None:
            self.b.data -= lr * self.b.grad
            self.b_history_grad = self.b.grad


class Sequential:
    def __init__(self, input=[]):
        self.model_list = input

    def forward(self, input):
        x = input
        for i in range(len(self.model_list)):
            x = self.model_list[i].forward(x)
        return x

    def step(self, lr=0.001):
        for i in range(len(self.model_list)):
            self.model_list[i].step(lr)

    def zero_grad(self):
        for i in range(len(self.model_list)):
            self.model_list[i].zero_grad()

    def backward(self, input=None):
        x = input
        for i in reversed(range(len(self.model_list))):
            x = self.model_list[i].backward(x)


class L2loss(Basic):
    def __init__(self, net: Basic):
        super(L2loss, self).__init__()
        self.net = net

    def forward(self, input, label):
        self.forward_value = (input - label.reshape(input.shape)).transpose(1, 2)
        return ((input - label) ** 2).mean()

    def backward(self, input=None):
        return self.net.backward(self.forward_value)


class CrossEntropy(L2loss):
    def __init__(self, net: Basic):
        super(CrossEntropy, self).__init__(net)

    def forward(self, input, label):
        self.forward_value = (-label.reshape(input.shape) * 1 / input + (1 - label.reshape(input.shape)) * 1 / (
                1 - input)).transpose(1, 2)
        return (-label.reshape(input.shape) * torch.log(input) - (1 - label.reshape(input.shape)) * torch.log(
            1 - input)).mean()

    def backward(self, input=None):
        super(CrossEntropy, self).backward()
