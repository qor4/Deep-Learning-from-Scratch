import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x + y)
print(x - y)
print(x * y)
print(x / y)

#배열 x와 y의 원소 수가 같다면 산술 연산은 각 원소에 대해서 행해진다.

x = np.array([1.0, 2.0, 3.0])
print(x / 2.0)

#넘파이 배열과 수치 하나(스칼라값)의 조합으로 된 산술 연산은 스칼라값과 넘파이 배열의 원소별로 한 번씩 수행된다.
#이를 브로드캐스트라고 한다.