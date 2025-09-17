import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
print("hello world")
class dog:
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def bark(self):
        print(f"{self.name},woof woof")
mydog=dog("sam",10)
mydog.bark()
print(mydog.age)
x=np.array([1,2,3,4])
print(x)
print(type(x))
y=np.array([1,2,3,4])
print(x+y)
z=np.array([[1,2,3,4],[5,6,7,8]])
print(z)
print(z.shape)
print(z.dtype)
print(x*z)
print(z[1][2])
m=np.arange(0,6,0.1)
n=np.sin(m)
plt.plot(m,n)
v=np.arange(0,6,0.1)
v1=np.cos(v)
plt.plot(m,n,label="sin")
plt.plot(v,v1,linestyle="--",label="cos")#绘制图案
plt.xlabel("x")
plt.ylabel("y")#添加标签
plt.title("sin and cos")
plt.legend()#显示图像
plt.show()#弹出窗口显示图像
img=imread(r'D:\vscode\torch\hymenoptera_data\train\ants\0013035.jpg')
print(f"图像形状: {img.shape}")
print(f"图像数据类型: {img.dtype}")
print(f"图像值范围: {img.min()} ~ {img.max()}")
plt.imshow(img)
plt.title("Ant Image")
plt.show()