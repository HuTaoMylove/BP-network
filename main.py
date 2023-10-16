import torch
import torch.nn as nn
from model import Linear, Sequential, Sigmoid, Tanh
from utils import load_data

if __name__ == "__main__":

    """
    设置随机种子
    """
    torch.manual_seed(2023)
    """
    读取训练数据
    """
    training_data, labels, gender = load_data()

    """
    数据归一化
    """
    # training_data = (training_data - training_data.mean(0, keepdim=True)) / training_data.std(0, keepdim=True)
    training_data = (training_data - training_data.min(0, keepdim=True)[0]) / (
            training_data.max(0, keepdim=True)[0] - training_data.min(0, keepdim=True)[0]) * 2 - 1
    boydata = training_data[gender]
    girldata = training_data[~gender]
    """
    定义网络架构
    """
    net = Sequential([Linear(5, 5), Linear(5, 1), Sigmoid()])

    for i in range(3000):
        bidx = torch.randperm(boydata.shape[0])[:24]
        gidx = torch.randperm(girldata.shape[0])[:24]
        label = torch.ones(48).reshape(-1, 1)
        label[24:, 0] = 0
        batch = torch.cat([boydata[bidx], girldata[gidx]], dim=0)
        net.zero_grad()
        out = net.forward(batch.unsqueeze(-1))
        loss = ((out - label) ** 2).mean()
        net.backward((out - label.reshape(out.shape)).transpose(1, 2))
        net.step(0.01)
        with torch.no_grad():
            out = net.forward(training_data.unsqueeze(-1))
            p = out > 0.5
            print((p * labels.reshape(p.shape) + ~p * (1 - labels.reshape(p.shape))).mean())
