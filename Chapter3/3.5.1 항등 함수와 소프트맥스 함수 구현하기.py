import numpy as np

def softmax(a):
    exp_a = np.exp(a)  # 지수 함수
    sum_exp_a = np.sum(exp_a)  # 지수 함수의 합
    y = exp_a / sum_exp_a

    return y

a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a) #지수 함수
print(exp_a)

sum_exp_a = np.sum(exp_a) #지수 함수의 합
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)