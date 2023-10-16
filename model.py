import torch
import copy


class Basic:
    def __init__(self):
        self.w = None
        self.b = None
        self.forward_value = None

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
        out = (1 - torch.exp(-2*input)) / (1 + torch.exp(-2*input))
        self.forward_value = copy.deepcopy(out.detach().clone())
        return out
    def backward(self, input):
        return input * (1-self.forward_value.transpose(1, 2)*(1-self.forward_value.transpose(1, 2)))



class Linear(Basic):
    def __init__(self, input, output, bias=True):
        super(Linear, self).__init__()
        assert type(input) is int
        assert type(output) is int
        self.w = 0.1 * torch.randn([output, input], dtype=torch.float32, requires_grad=True)
        if bias:
            self.b = 0.1 * torch.randn([output, 1], dtype=torch.float32, requires_grad=True)

    def zero_grad(self):
        self.w.grad = torch.zeros_like(self.w)
        if self.b is not None:
            self.b.grad = torch.zeros_like(self.b)

    def forward(self, input):
        self.batchsize, self.input_channel = input.shape[0], input.shape[1]
        w = self.w.repeat(self.batchsize, 1, 1)
        if self.b is not None:
            b = self.b.repeat(self.batchsize, 1, 1)

        out = torch.bmm(w, input)
        self.forward_value = copy.deepcopy(input.detach().clone())
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
        if self.b is not None:
            self.b.data -= lr * self.b.grad


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
