import numpy as np

a = np.array([1010, 1000, 990])
print(np.exp(a) / np.sum(np.exp(a))) #[nan nan nan] overflow로 계산이 되지 않음 (nan = not a number)
c = np.max(a) #1010
print(a - c)
print(np.exp(a - c) /np.sum(np.exp(a - c)))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(np.exp(a - c))
    y = exp_a / sum_exp_a

    return y