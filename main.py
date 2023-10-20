import matplotlib.pyplot as plt
import torch
import nn
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import numpy as np

import optims
import utils

torch.manual_seed(2023)


def train(batchsize=56, KFold_num=10, lr=0.2, epoch=1500):
    """设置超参数
    :param batchsize: 一次反向传播的样本数量
    :param KFold_num: kfold的数量
    :param lr: 学习率
    :param epoch: 训练次数
    """

    """******************
    读取训练数据及数据归一化
    ********************"""

    training_data, labels, gender = utils.load_data()
    training_data = (training_data - training_data.mean(0, keepdim=True)) / training_data.std(0, keepdim=True)
    training_data = (training_data - training_data.min(0, keepdim=True)[0]) / (
            training_data.max(0, keepdim=True)[0] - training_data.min(0, keepdim=True)[0])
    boydata = training_data[gender]
    girldata = training_data[~gender]

    """******************
    设置kfold数据集划分并对
    ********************"""
    kf = KFold(n_splits=KFold_num, shuffle=False)
    res = 0
    train_losses = np.zeros(epoch)
    test_losses = np.zeros(epoch)
    train_acc = np.zeros(epoch)
    test_acc = np.zeros(epoch)

    """******************
    基于数据划分开始训练
    ********************"""
    real_label = []
    perdict_label = []
    for ((boy_train, boy_test), (girl_train, girl_test)) in zip(kf.split(boydata), kf.split(girldata)):
        boytrain = boydata[boy_train]
        girltrain = girldata[girl_train]
        testdata = torch.cat([boydata[boy_test], girldata[girl_test]], dim=0)
        test_label = torch.ones(testdata.shape[0]).reshape(-1, 1)
        test_label[boydata[boy_test].shape[0]:] = 0

        """******************
        设置网络结构
        
        Sequential中设置list，来搭建期望的网络
        可选： Linear，Sigmoid，Tanh
        loss为损失函数的选择
        可选： L2loss,CrossEntropy
        optim为网络的优化器，支持动量和weight_decay
        可选： SGD,SGDM
        ********************"""
        batchsize = 48
        net = nn.Sequential(
            [nn.Linear(input=5, output=5, bias=True),
             nn.Linear(input=5, output=5, bias=True),
             nn.Linear(input=5, output=5, bias=True),
             nn.Linear(input=5, output=1, bias=True),
             nn.Sigmoid()])
        loss = nn.CrossEntropy(net)
        optim = optims.SGDM(net=net, lr=lr, weight_decay=0.001, momentum=0.9)

        best_acc = 0
        for i in range(epoch):
            bidx = torch.randperm(boytrain.shape[0])[:batchsize // 2]
            gidx = torch.randperm(girltrain.shape[0])[:batchsize // 2]
            label = torch.ones(batchsize).reshape(-1, 1)
            label[batchsize // 2:, 0] = 0
            batch = torch.cat([boydata[bidx], girldata[gidx]], dim=0)

            """
            训练环节主要流程
            （1）优化器将net梯度置零
            （2）net计算出结果
            （3）将预测结果与标签输入损失函数进行计算
            （4）从loss将梯度反向传播
            （5）优化器执行优化，参数根据对应梯度修改
            """
            optim.zero_grad()
            out = net.forward(batch.unsqueeze(-1))
            loss_value = loss.forward(out, label)
            train_losses[i] += loss_value
            loss.backward()
            optim.step()
            p = out > 0.5
            acc = (p * label.reshape(p.shape) + ~p * (1 - label.reshape(p.shape))).mean()
            train_acc[i] += acc
            """
            测试环节，测试准确率
            """
            out = net.forward(testdata.unsqueeze(-1))
            p = out > 0.5
            acc_t = (p * test_label.reshape(p.shape) + ~p * (1 - test_label.reshape(p.shape))).mean()
            test_losses[i] += loss.forward(out, test_label)
            best_acc = max(best_acc, acc_t)
            test_acc[i] += acc_t
            if i == epoch - 1:
                real_label.append(test_label.numpy())
                perdict_label.append(p.squeeze(-1).numpy().astype(np.float32))
        res += best_acc.data
        # print(acc)

    """
      绘图环节，计算SE, SP, ACC
    """
    # 绘制ROC曲线
    plt.figure(figsize=[16, 6])
    plt.subplot(1, 3, 1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC曲线图')
    for step in range(len(real_label)):
        r, p = real_label[step], perdict_label[step]
        fpr, tpr, _ = roc_curve(r, p)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label=f'area = {round(roc_auc, 3)},fold={step + 1}')
    real_label = np.concatenate(real_label, axis=0)
    perdict_label = np.concatenate(perdict_label, axis=0)
    fpr, tpr, _ = roc_curve(real_label, perdict_label)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='orange', lw=3, label=f'area = {round(roc_auc, 3)},fold=mean')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")

    # 计算SE，SP，ACC指标
    print("BEST_ACC:", res / KFold_num)
    SE, SP, ACC = utils.accuracy_calculation(real_label, perdict_label)
    print("SE:", SE)
    print("SP:", SP)
    print("ACC:", ACC)

    # 绘制损失和准确率图
    train_losses = train_losses / KFold_num
    test_losses = test_losses / KFold_num
    train_acc = train_acc / KFold_num
    test_acc = test_acc / KFold_num
    plt.subplot(1, 3, 2)
    plt.xlabel('训练步数')
    plt.ylabel('损失值')
    plt.title('训练损失图')
    plt.plot(np.arange(epoch) + 1, train_losses, color='red', linestyle='-', label='训练损失')
    plt.plot(np.arange(epoch) + 1, test_losses, color='blue', linestyle='-', label='测试损失')
    plt.legend(loc="upper right")
    plt.subplot(1, 3, 3)
    plt.xlabel('训练步数')
    plt.ylabel('准确率')
    plt.title('准确率图')
    plt.plot(np.arange(epoch) + 1, train_acc, color='red', linestyle='-', label='训练准确率')
    plt.plot(np.arange(epoch) + 1, test_acc, color='blue', linestyle='-', label='测试准确率')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    train(KFold_num=10, batchsize=48)
