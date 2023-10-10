import torch


class BasicModel:
    def __init__(self):
        self.w = None
        self.bias = None
        self.mean = None
        self.var = None

    def forward(self, input):
        pass

    def backward(self, input):
        pass


class Linear(BasicModel):
    def __init__(self, input, output, bias=True):
        super(Linear, self).__init__()
        assert type(input) is int
        assert type(output) is int
        self.w = torch.rand([output, input], dtype=torch.float32, requires_grad=True)
        if bias:
            self.b = torch.rand([output, 1], dtype=torch.float32, requires_grad=True)
        self.forward_value = None

    def zero_grad(self):
        self.w.grad = torch.ones_like(self.w)
        if self.b is not None:
            self.b.grad = torch.ones_like(self.b)

    def forward(self, input):
        self.batchsize, self.input_channel = input.shape[0], input.shape[1]
        w = self.w.repeat(self.batchsize, 1, 1)
        b = self.b.repeat(self.batchsize, 1, 1)
        out = torch.bmm(w, input.unsqueeze(-1))
        self.forward_value = input
        return out + b

    def backward(self, input=None):
        if input is None:
            self.w.grad = self.forward_value.mean(0, keepdim=True) + self.w.grad
            self.b.grad = torch.ones_like(self.b.grad)
        else:
            self.w.grad = torch.bmm(input, self.forward_value.unsqueeze(-2)).mean(0) + self.w.grad
            self.b.grad = input.mean(0)

    def step(self, lr=0.001):
        self.w.data -= lr * self.w.grad
        self.b.data -= lr * self.b.grad
