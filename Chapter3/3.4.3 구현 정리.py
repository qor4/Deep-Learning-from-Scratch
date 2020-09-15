import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) #np.exp(-x)는 브로드캐스트 기능으로 넘파이 배열을 반환

def identity_function(x):
    return x

#가중치와 편향 초기화, 딕셔너리 변수 network에 저장
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

#입력 신호를 출력으로 변환하는 처리 과정
#함수 이름 forward인 이유는 신호가 순방향(입력에서 출력 방향)인 순전파임을 알리기 위해서
#역방향은 backward
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)

#신경망은 분류 회귀 모두에 이용할 수 있다.
#출력층에서 사용하는 활성화 함수는 일반적으로 회귀 - 항등 함수, 분류 - 소프트맥스 함수 사용



