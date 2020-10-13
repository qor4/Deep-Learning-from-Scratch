import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] #원-핫 인코딩 정답은 2

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] #2일 확률이 가장 높다고 추정
print(cross_entropy_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0] #7일 확률이 가장 높다고 추정
print(cross_entropy_error(np.array(y), np.array(t)))

#7보다 2일때 출력값이 더 작음 -> 작을 수록 오차가 작다.