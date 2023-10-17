import torch
import nn
from utils import load_data
import optims

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
    net = nn.Sequential([nn.Linear(5, 12), nn.Linear(12, 12), nn.Linear(12, 1), nn.Sigmoid()])
    loss = nn.CrossEntropy(net)
    optim = optims.SGD(net, 0.01, 0.001)
    for i in range(2000):
        bidx = torch.randperm(boydata.shape[0])[:24]
        gidx = torch.randperm(girldata.shape[0])[:24]
        label = torch.ones(48).reshape(-1, 1)
        label[24:, 0] = 0
        batch = torch.cat([boydata[bidx], girldata[gidx]], dim=0)
        optim.zero_grad()
        out = net.forward(batch.unsqueeze(-1))
        loss_value = loss.forward(out, label)
        loss.backward()
        optim.step()
        with torch.no_grad():
            out = net.forward(training_data.unsqueeze(-1))
            p = out > 0.5
            print((p * labels.reshape(p.shape) + ~p * (1 - labels.reshape(p.shape))).mean())
