import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(np.int)

x = np.array([-1.0, 1.0, 2.0])
print(x)
y = x > 0
print(y)

y = y.astype(np.int) #bool을 int로 변환하면 True는 1, False는 0으로 변환
print(y)