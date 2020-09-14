import numpy as np

def relu(x):
    return np.maximum(0, x) #넘파이의 maximum - 두 입력 중 큰 값을 선택해 반환

#ReLU (렐루)
#입력이 0을 넘으면 그 입력을 그대로 출력하고, 0 이하이면 0을 출력하는 함수