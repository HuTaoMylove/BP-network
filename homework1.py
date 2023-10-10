import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pylab import mpl

# 设置字体
mpl.rcParams['font.sans-serif'] = ["SimHei"]
mpl.rcParams['axes.unicode_minus'] = False


def data_washing(arr):
    # 使用均值填充nan
    mask = np.isnan(arr)
    mean = np.mean(arr[~mask])
    arr[mask] = mean
    return arr


def hist(height, gender):
    plt.suptitle("身高统计直方图")
    bins = np.linspace(130, 200, 20)
    plt.hist([height[gender], height[~gender]], bins, label=['男生身高', '女生身高'])
    plt.legend(loc='upper left')
    plt.show()


def BE(prior_mean, prior_var, var, data):
    # 贝叶斯估计公式
    return (prior_var * np.sum(data) + var * prior_mean) / (data.shape[0] * prior_var + var)


def G(cov1: np.array, mean1: np.array, prior1: np.array, cov2: np.array, mean2: np.array, prior2: np.array,
      x: np.array):
    # 决策面公式
    d = 0.5 * np.matmul(x - mean1, np.matmul(np.linalg.inv(cov1), (x - mean1).T)) - \
        0.5 * np.matmul(x - mean2, np.matmul(np.linalg.inv(cov2), (x - mean2).T))
    d = np.diagonal(d)
    c = 0.5 * np.log(np.linalg.det(cov1) / np.linalg.det(cov2))
    p = np.log(prior1 / prior2)
    return d + c - p


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
    '''
    作业一，绘制直方图
    '''
    hist(height, gender)

    '''
    作业二，最大似然估计
    '''
    male_height_mean = np.mean(height[gender])
    male_height_var = np.var(height[gender])
    famale_height_mean = np.mean(height[~gender])
    famale_height_var = np.var(height[~gender])
    print(f'男性身高均值：{male_height_mean}，方差：{male_height_var}\n女性身高均值：{famale_height_mean}，方差：{famale_height_var}')

    '''
    作业三，贝叶斯估计
    假设已知男生身高方差为40，女生身高方差为30
    男生均值先验为172.6，女生均值先验为160.6
    均值概率方差先验均为30
    '''
    prior_male_mean = 172.6
    prior_male_var = 30
    prior_famale_mean = 160.6
    prior_famale_var = 30
    male_var = 40
    famale_var = 30

    BE_male_mean = BE(prior_male_mean, prior_male_var, male_var, height[gender])
    BE_famale_mean = BE(prior_famale_mean, prior_famale_var, famale_var, height[~gender])
    print('基于贝叶斯估计的男生身高均值为', BE_male_mean)
    print('基于贝叶斯估计的女生身高均值为', BE_famale_mean)
    '''
    作业四，分类器设计
    '''
    male_cov = np.cov(np.concatenate([height[gender].reshape(1, -1), weight[gender].reshape(1, -1)]))
    male_mean = np.array([np.mean(height[gender]), np.mean(weight[gender])]).reshape(1, 2)
    print('男生均值', '\n', male_mean, '\n', '男生协方差矩阵', '\n', male_cov)
    famale_cov = np.cov(np.concatenate([height[~gender].reshape(1, -1), weight[~gender].reshape(1, -1)]))
    famale_mean = np.array([np.mean(height[~gender]), np.mean(weight[~gender])]).reshape(1, 2)
    print('女生均值', '\n', famale_mean, '\n', '女生协方差矩阵', '\n', famale_cov)

    prior_male = np.mean(gender)
    prior_famale = np.mean(~gender)
    x1 = np.array([165., 50.]).reshape(1, 2)
    x2 = np.array([175., 55.]).reshape(1, 2)

    judge = G(male_cov, male_mean, prior_male, famale_cov, famale_mean, prior_famale, x1)
    if judge > 0:
        print(' [165., 50.] 是女生')
    else:
        print(' [165., 50.] 是男生')
    judge = G(male_cov, male_mean, prior_male, famale_cov, famale_mean, prior_famale, x2)
    if judge > 0:
        print(' [175., 55.] 是女生')
    else:
        print(' [175., 55.] 是男生')

    # 绘制决策面图像
    xmin = np.min(height)
    xmax = np.max(height)
    ymin = np.min(weight)
    ymax = np.max(weight)
    xx = np.linspace(xmin, xmax, 100)
    yy = np.linspace(ymin, ymax, 100)
    X, Y = np.meshgrid(xx, yy)
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    mask = abs(G(male_cov, male_mean, prior_male, famale_cov, famale_mean, prior_famale,
                 np.concatenate([X, Y], axis=1))) < 0.2

    fig, ax = plt.subplots()
    ax.scatter(height[gender], weight[gender], c=np.array([1, 0, 0]), s=10, label='男生')
    ax.scatter(height[~gender], weight[~gender], c=np.array([0, 1, 0]), s=10, label='女生')
    ax.scatter(X[mask], Y[mask], c=np.array([0, 0, 1]), label='决策面')

    ax.legend()
    plt.title("决策面及样本分布图")
    plt.xlabel("身高")
    plt.ylabel("体重")
    plt.show()
