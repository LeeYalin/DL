# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:07:29 2020

@author: liyalin
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

def lossFunction(x):
    return (x-2.5)**2-1

# 在-1到6的范围内构建140个点
plot_x = np.linspace(-1,6,141)
# plot_y 是对应的损失函数值
plot_y = lossFunction(plot_x)

plt.plot(plot_x,plot_y)
plt.show()

def dLF(theta):
    return derivative(lossFunction, theta, dx=1e-6)

def lossFunction(x):
    try:
        return (x-2.5)**2-1
    except:
        return float('inf')
    
    
def gradient_descent(initial_theta, eta, n_iters, epsilon=1e-6):
    theta = initial_theta
    theta_history.append(theta)
    i_iters = 0
    while i_iters < n_iters:
        # 每一轮循环后，要求当前这个点的梯度是多少
        gradient = dLF(theta)
        last_theta = theta
        # 移动点，沿梯度的反方向移动步长eta
        theta = theta - eta * gradient
        theta_history.append(theta)
        # 判断theta是否达到损失函数最小值的位置
        if(abs(lossFunction(theta) - lossFunction(last_theta)) < epsilon):
            break
        i_iters += 1
        

def plot_theta_history():
    plt.plot(plot_x,plot_y)
    plt.plot(np.array(theta_history), lossFunction(np.array(theta_history)), color='red', marker='o')
    plt.show()
        
        
eta=0.1
theta_history = []
gradient_descent(0., eta, 1000)
plot_theta_history()
print("梯度下降查找次数：",len(theta_history))
