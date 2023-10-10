import torch
import pandas as pd
from homework1 import data_washing
import numpy as np
from model import Linear

if __name__ == "__main__":
    # 数据读取
    data = pd.read_excel('2023模式识别数据集汇总.xls')
    shoes = data_washing(data['鞋码'].values)
    height = data_washing(data['身高'].values)
    weight = data_washing(data['体重'].values)
    race = data_washing(data['50米'].values)
    lungs = data_washing(data['肺活量'].values)
    gender = data['性别'].values
    gender = gender > 0

    training_data = torch.tensor(torch.from_numpy(
        np.concatenate([shoes[..., None], height[..., None], weight[..., None], race[..., None], lungs[..., None]],
                       axis=-1)).clone().detach().requires_grad_(True), dtype=torch.float32)
    training_data=(training_data-training_data.mean(0,keepdim=True))/training_data.var(0,keepdim=True)
    label = torch.from_numpy(gender).to(torch.float32)
    net = Linear(5, 1)

    for i in range(200):
        net.zero_grad()
        out = torch.sigmoid(net.forward(training_data))
        print(torch.mean((out - label.reshape(out.shape))**2))
        net.backward((out - label.reshape(out.shape))*out*(1-out))
        net.step()