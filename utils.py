import pandas as pd
from homework1 import data_washing
import torch
import numpy as np


def load_data():
    data = pd.read_excel('2023模式识别数据集汇总.xls')
    shoes = data_washing(data['鞋码'].values)
    height = data_washing(data['身高'].values)
    weight = data_washing(data['体重'].values)
    race = data_washing(data['50米'].values)
    lungs = data_washing(data['肺活量'].values)
    gender = data['性别'].values
    gender = gender > 0

    torch.manual_seed(2023)
    training_data = torch.tensor(torch.from_numpy(
        np.concatenate([shoes[..., None], height[..., None], weight[..., None], race[..., None], lungs[..., None]],
                       axis=-1)).clone().detach(), dtype=torch.float32)

    height_idx = height < 200
    weight_idx = (40 < weight) * (weight < 100)
    lungs_idx = lungs > 200
    training_data = training_data[height_idx * weight_idx * lungs_idx]
    gender = gender[height_idx * weight_idx * lungs_idx]
    labels = torch.from_numpy(gender).to(torch.float32).reshape(-1, 1)
    return training_data, labels,gender
