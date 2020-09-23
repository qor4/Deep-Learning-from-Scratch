import numpy as np
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(np.exp(a - c))
    y = exp_a / sum_exp_a

    return y
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)

print(y)
print(np.sum(y))
#소프트맥스 함수의 출력은 0~1.0 사이의 실수, 또한 소프트맥스 함수 출력의 총합은 1 -> 소프트맥스 함수의 출력을 '확률'로 해석할 수 있음