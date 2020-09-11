import numpy as np

A = np.array([[1, 2],[3, 4]])
print(A)
print(A.shape) #행렬의 형상
print(A.dtype) #행렬에 담긴 원소의 자료형

B = np.array([[3, 0], [0, 6]])
print(A + B)
print(A * B)
#형상이 같은 행렬끼리면 행렬의 산술 연산도 대응하는 원소별로 계산
#행렬과 스칼라값의 산술 연산은 배열과 마찬가지로 브로드캐스트 기능 작동

print(A)
print(A * 10)