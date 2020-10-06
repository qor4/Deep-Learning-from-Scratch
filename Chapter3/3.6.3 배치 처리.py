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

def get_data(): #MNIST 데이터셋을 얻고
    (x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, flatten=True, one_hot_label=False) #normalize=True는 0~255 범위인 픽셀의 값을 0.0~1.0의 범위로 변환 -> 정규화, 신경망의 입력 데이터에 특정 변환을 가하는 것 -> 전처리
    #입력 이미지에 대한 전처리 작업으로 정규화 수행
    return x_test, t_test

def init_network(): #네트워크 생성
    with open("sample_weight.pkl", 'rb') as f: #sample_weight.pkl에는 '학습된 가중치 매개변수'가 가중치와 편향 매개변수가 딕셔너리 변수로 저장되어 있음
        network = pickle.load(f)
    return network

x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']

print(x.shape)
print(x[0].shape)
print(W1.shape)
print(W2.shape)
print(W3.shape)
#784 784X50 50X100 100X10 10으로 행렬곱 가능
#만약 데이터 100개를 묶어 연산한다면 100X784 784X50 50X100 100X10 100X10으로 100장 분량 데이터가 나옴

#배치 = 묶음
#이전과 다르게 배치 처리로 구현
x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size): #range에 인수가 3개라면 0에서 len(x)-1까지 batch_size 간격으로 증가하는 리스트 반환
    x_batch = x[i:i+batch_size] #입력 데이터를 묶음, i부터 i+batch_size번째까지의 데이터를 묶음 (batch_size가 100이기 때문에 100장씩 묶음)
    y_batch = predict(network, x_batch) #[0.2, 0.3, 0.5 ... 0.03]와 같다면 0일 확률 0.2, 1일 확률 0.3 ...으로 해석
    p = np.argmax(y_batch, axis=1) #argmax -> 최댓값의 인덱스를 가져옴, axis -> 100X10의 배열 중 1번째 차원을 구성하는 각 원소에서 최댓값의 인덱스를 찾도록 함
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy : " + str(float(accuracy_cnt) / len(x))) #맞힌 숫자에 전체 이미지 숫자로 나눠 정확도를 구함