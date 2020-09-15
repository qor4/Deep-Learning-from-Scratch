import numpy as np

A = np.array([[1, 2], [3, 4]])
print(A.shape)
B = np.array([[5, 6], [7, 8]])
print(B.shape)
print(np.dot(A, B)) #np.dot() - 행렬의 곱, 입력이 1차원 배열이면 벡터, 2ㅏ원 배열이면 행렬 곱 계산

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape)
print(np.dot(A, B))

#첫번째 피연산자의 열의 수와 두번째 피연산자의 행의 수가 같아야 한다.
#다르면 오류 출력
#np.dot() 연산 결과 배열은 첫번째 피연산자의 행의 수만큼 행이, 두번째 피연산자의 열의 수만큼 열이 생성된다.