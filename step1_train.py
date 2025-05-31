# -*-coding:utf-8 -*-

from model.unet_model import ResUNet
from utils.dataset import Dateset_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


def train_net(net, device, data_path, epochs=100, batch_size=1, lr=0.00001):
    '''
    :param net: 语义分割网络
    :param device: 网络训练所使用的设备
    :param data_path: 数据集的路径
    :param epochs: 训练的轮数
    :param batch_size: 批次大小
    :param lr: 学习率
    :return:
    '''
    # 加载数据集
    dataset = Dateset_Loader(data_path)
    per_epoch_num = len(dataset) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    optimizer = optim.Adam(net.parameters(),lr=lr,betas=(0.9, 0.999),eps=1e-08, weight_decay=1e-08,amsgrad=False)
    criterion = nn.MSELoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 开始训练
    loss_record = []
    with tqdm(total=epochs*per_epoch_num) as pbar:
        for epoch in range(epochs):
            # 训练模式
            net.train()
            # 按照batch_size开始训练
            for image, label in train_loader:
                optimizer.zero_grad()
                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = net(image)
                 # MSE损失函数需要先将预测结果通过sigmoid函数转换到0-1之间
                pred = torch.sigmoid(pred)
                loss = criterion(pred, label)
                pbar.set_description("Processing Epoch: {} Loss: {}".format(epoch+1, loss))
                if loss<best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'best_model_1_min.pth')
                loss.backward()
                optimizer.step()
                pbar.update(1)
            # print(loss.item())
            loss_record.append(loss.item())

    # 绘制loss折线图
    plt.figure()
    # 绘制折线图
    plt.plot([i+1 for i in range(0, len(loss_record))], loss_record)
    # 添加标题和轴标签
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('/data1/limuci/code/cvlab/unet_42-drive/results/training_loss_1_min.png')

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    torch.cuda.set_device(2)
    device = torch.device(f'cuda:{2}' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = ResUNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path="DRIVE-SEG-DATA"
    train_net(net, device, data_path, epochs=100, batch_size=4)  # 开始训练，如果你GPU的显存小于4G，这里只能使用CPU来进行训练。
