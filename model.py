import torch


class BasicModel:
    def __init__(self):
        self.w = None
        self.bias = None
        self.forward_value = None

    def forward(self, input):
        pass

    def backward(self, input):
        pass


class BasicActivation:
    def __init__(self):
        pass

    def forward(self, input):
        pass

    def backward(self, input):
        pass

    def zero_grad(self):
        pass

    def step(self, lr=0.001):
        pass


class BasicNorm:
    def __init__(self):
        self.mean = None
        self.var = None
        self.forward_value = None

    def forward(self, input):
        pass

    def backward(self, input):
        pass

    def step(self, lr=0.001):
        pass


class Sigmoid(BasicActivation):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input):
        self.forward_value = 1 / (1 + torch.exp(-input))
        return self.forward_value

    def backward(self, input):
        return input * self.forward_value.transpose(1, 2) * (1 - self.forward_value.transpose(1, 2))


class Linear(BasicModel):
    def __init__(self, input, output, bias=True):
        super(Linear, self).__init__()
        assert type(input) is int
        assert type(output) is int
        self.w = torch.rand([output, input], dtype=torch.float32, requires_grad=True)
        if bias:
            self.b = torch.rand([output, 1], dtype=torch.float32, requires_grad=True)

    def zero_grad(self):
        self.w.grad = torch.ones_like(self.w)
        if self.b is not None:
            self.b.grad = torch.ones_like(self.b)

    def forward(self, input):
        self.batchsize, self.input_channel = input.shape[0], input.shape[1]
        w = self.w.repeat(self.batchsize, 1, 1)
        b = self.b.repeat(self.batchsize, 1, 1)
        out = torch.bmm(w, input)
        self.forward_value = input
        return out + b

    def backward(self, input=None):
        if input is None:
            self.w.grad = self.forward_value.mean(0, keepdim=True) + self.w.grad
            self.b.grad = torch.ones_like(self.b.grad)
            return self.w.repeat(self.batchsize, 1, 1)
        else:
            self.w.grad = torch.bmm(input.transpose(1, 2), self.forward_value.transpose(1, 2)).mean(0) + self.w.grad
            self.b.grad = input.mean(0).T
            return torch.bmm(input, self.w.repeat(self.batchsize, 1, 1))

    def step(self, lr=0.001):
        self.w.data -= lr * self.w.grad
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
