import torch
from torchvision import datasets,transforms
import numpy as np
from PIL import Image
import sys, os
import pickle
#x：代表 ​​输入数据（特征
#t是数据对应的​​正确答案​​或​​目标值​​
train_dataset=datasets.MNIST(root="./data",train=True,download=True,transform=transforms.ToTensor())
test_dataset=datasets.MNIST(root="./data",train=False,download=True,transform=transforms.ToTensor())
x_train = train_dataset.data.numpy() # 形状为 (60000, 28, 28)
t_train = train_dataset.targets.numpy() # 形状为 (60000,)

x_test = test_dataset.data.numpy() # 形状为 (10000, 28, 28)
t_test = test_dataset.targets.numpy() # 形状为 (10000,)
"""
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
"""
def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))# 将numpy数组转为PIL图像
    pil_img.show()
def get_data():
    """
    加载MNIST数据集并返回测试数据
    
    Returns:
        tuple: 包含测试数据的元组
            - x_test: 测试图像数据，形状为(样本数, 28, 28)的numpy数组
            - t_test: 测试标签数据，形状为(样本数,)的numpy数组
    """
    return x_test,t_test
def init_network():#init_network()会读入保存在pickle文件sample_weight.pkl中的学习到的权重参数
    with open(r"D:\vscode\深度学习入门\sample_weight.pkl", "rb") as f:
        network=pickle.load(f)
    return network
def predict(network, x):
    """
    使用三层神经网络进行预测
    
    参数:
        network: 包含网络参数的字典,包括权重W1,W2,W3和偏置b1,b2,b3
        x: 输入数据
        
    返回:
        y: 经过softmax处理后的输出结果
    """
    # 提取网络参数
    # 提取网络参数
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    # 全部使用 numpy 进行计算
    a1 = np.dot(x, W1) + b1
    z1 = 1 / (1 + np.exp(-a1))  # 使用 numpy 的 sigmoid 函数
    a2 = np.dot(z1, W2) + b2
    z2 = 1 / (1 + np.exp(-a2))  # 使用 numpy 的 sigmoid 函数
    a3 = np.dot(z2, W3) + b3
    
    # 手动实现 softmax 以避免维度问题
    exp_a3 = np.exp(a3 - np.max(a3))  # 减去最大值以提高数值稳定性
    y = exp_a3 / np.sum(exp_a3)
    
    return y
"""
img = x_train[0]
label=t_train[0]
print(label)
img=img.reshape(28,28)# flatten=True时读入的图像是以一列(一维)NumPy数组的形式保存的。因此,显示图像时,需要把它变为原来的28像素 x 28像素的形状。
img_show(img)
"""
x,t=get_data()
x = x.reshape(-1, 784)  # 从 (10000, 28, 28) 变为 (10000, 784)
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  # 现在 y 是 numpy 数组，可以直接使用 np.argmax
    if p == t[i]:
        accuracy_cnt += 1
print("Accuracy:"+str(float(accuracy_cnt)/len(x)))