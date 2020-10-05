import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import pickle

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(np.exp(a - c))
    y = exp_a / sum_exp_a
    return y

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) #np.exp(-x)는 브로드캐스트 기능으로 넘파이 배열을 반환

def get_data(): #MNIST 데이터셋을 얻고
    (x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, flatten=True, one_hot_label=False) #normalize=True는 0~255 범위인 픽셀의 값을 0.0~1.0의 범위로 변환 -> 정규화, 신경망의 입력 데이터에 특정 변환을 가하는 것 -> 전처리
    #입력 이미지에 대한 전처리 작업으로 정규화 수행
    return x_test, t_test

def init_network(): #네트워크 생성
    with open("sample_weight.pkl", 'rb') as f: #sample_weight.pkl에는 '학습된 가중치 매개변수'가 가중치와 편향 매개변수가 딕셔너리 변수로 저장되어 있음
        network = pickle.load(f)
    return network

def predict(network, x): #각 레이블의 확률을 넘파이 배열로 반환
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0

for i in range(len(x)): #x에 저장된 이미지 데이터를 하나씩 꺼내 predict() 함수로 분류
    y = predict(network, x[i]) #[0.2, 0.3, 0.5 ... 0.03]와 같다면 0일 확률 0.2, 1일 확률 0.3 ...으로 해석
    p = np.argmax(y) #확률이 가장 높은 원소의 인덱스를 얻는다. == 예측 결과
    if p == t[i]: #신경망이 예측한 수와 정답을 비교해서 맞을때마다 cnt++
        accuracy_cnt += 1

print("Accuracy : " + str(float(accuracy_cnt) / len(x))) #맞힌 숫자에 전체 이미지 숫자로 나눠 정확도를 구함