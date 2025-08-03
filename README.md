# DRIVE 数据集血管分割项目

本项目为《计算机视觉》课程结课作业，主要实现了在 DRIVE 数据集上的血管分割任务，包含数据预处理、模型训练、预测与评估等完整流程。

## 项目结构

- `model`：UNet 模型相关代码，包括训练、预测、评估等脚本。
- `result`：实验相关代码与可视化结果。
- `tools`：一些预处理方法。

## 数据集说明

本项目使用 DRIVE 数据集，目录结构如下：

- `DRIVE-SEG-DATA/Training_Images/`：训练图片
- `DRIVE-SEG-DATA/Training_Labels/`：训练标签
- `DRIVE-SEG-DATA/Test_Images/`：测试图片
- `DRIVE-SEG-DATA/Test_Labels/`：测试标签

## 环境依赖

建议使用 Python 3.8 及以上版本，主要依赖如下：

- torch
- torchvision
- numpy
- opencv-python
- matplotlib
- tqdm
- wandb

可通过如下命令安装依赖：

```bash
pip install -r UNet/requirements.txt
```

## 运行方法

### 1. 数据预处理

可使用 `utils/`文件夹下面的预处理方式将 tif 格式图片转换为 png。

### 2. 模型训练

运行：

```bash
python step1_train.py
```

训练完成后，模型权重会保存在 `UNet/checkpoints/` 或根目录下。

### 3. 模型预测

在 `UNet/` 目录下运行：

```bash
python step2_predict.py
```

预测结果会保存在 `results/`。


## 结果展示


![损失函数](https://github.com/Li-MoCi/cvlab/blob/main/results/training_loss_1.png)


## 致谢

感谢《计算机视觉》课程的指导与 DRIVE 数据集的开放。

---

如有问题欢迎 issue 或联系作者。
