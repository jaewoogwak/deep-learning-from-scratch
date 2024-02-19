import sys
import os
path = os.getcwd()
parent = os.path.join(path, os.pardir)
sys.path.append(os.path.abspath(parent))

from common.gradient import function_2, numerical_gradient
from common.functions import softmax, cross_entropy_error
import numpy as np


# function_2의 최솟값 구하기, 초기 값은 -3.0, 4.0으로 주어짐
# init_x = np.array([-3.0, 4.0])
# print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))


# 4.4.2 신경망에서의 기울기
class simpleNet:
    """docstring for simpleNet"""
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()
print(net.W)  # 가중치 매개변수(랜덤)
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))  # 최댓값의 인덱스

t = np.array([0, 0, 1])  # 정답 레이블
print(net.loss(x, t))


def f(W):
    return net.loss(x, t)


dW = numerical_gradient(f, net.W)
print(dW)




