import numpy as np
import matplotlib.pyplot as plt
def fun(x):
    return np.array(x>0,dtype=int)
m = np.arange(-5, 5, 0.1)
n=fun(m)
plt.ylim(-0.1,1.1)# 指定y轴的范围
def sigmoid(x):# sigmoid函数
    return 1/(1+np.exp(-x))
x=np.arange(-5.0,5.0,0.1)
y=sigmoid(x)
plt.plot(x,y)
plt.plot(m,n,linestyle="--")
# plt.ylim(-1.0,1.1)
plt.show()
def relu(x):# relu函数
 return np.maximum(0, x)
#多维数组的运算
A = np.array([1, 2, 3, 4])
print(np.ndim(A))#数组的维数可以通过np.dim()函数获得
print(np.shape(A))#数组的形状可以通过np.shape()函数获得
B = np.array([[1,2], [3,4], [5,6]])
print(B)
'''输出：
[[1 2]
 [3 4]
 [5 6]]
'''
print(np.ndim(B))#输出2
print(B.shape)#输出(3, 2)
C=np.array([1,2])
print(np.dot(B,C))
'''
矩阵的乘积是通过左边矩阵的行（横向）和右边矩阵的列（纵
向）以对应元素的方式相乘后再求和而得到的
输出
array([[19, 22],
 [43, 50]])
 实现该神经网络时,要注意X、W、Y的形状,特别是X和W的对应
维度的元素个数是否一致，这一点很重要。
>>> X = np.array([1, 2])
>>> X.shape
(2,)
>>> W = np.array([[1, 3, 5], [2, 4, 6]])
>>> print(W)
[[1 3 5]
 [2 4 6]]
>>> W.shape56 第3章神经网络
(2, 3)
>>> Y = np.dot(X, W)
>>> print(Y)
[ 5 11 17]
'''
#3层神经网络的实现代码
def init_network():
 network = {}
 network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
 network['b1'] = np.array([0.1, 0.2, 0.3])
 network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
 network['b2'] = np.array([0.1, 0.2])
 network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
 network['b3'] = np.array([0.1, 0.2])
 return network
def forward(network, x):
 W1, W2, W3 = network['W1'], network['W2'], network['W3']
 b1, b2, b3 = network['b1'], network['b2'], network['b3']
 a1 = np.dot(x, W1) + b1
 z1 = sigmoid(a1)
 a2 = np.dot(z1, W2) + b2
 z2 = sigmoid(a2)
 a3 = np.dot(z2, W3) + b3
 y = a3
 return y
network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y) # [ 0.31682708 0.69627909]