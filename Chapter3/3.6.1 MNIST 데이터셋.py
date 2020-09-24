import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = \
load_mnist(flatten=True, normalize=False)
#"(훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)" 형식으로 반환, 인수로는 normalize, flatten, one_hot_laber이고 세 인수 모두 bool 값
#normalize는 입력 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지를 정함 -> False = 픽셀 값을 원래 값 그대로 0~255 사이의 값 유지
#flatten은 입력 이미지를 평탄하게, 1차원 배열로 만들지를 정함 -> False = 1 X 28 X 28의 3차원 배열, True = 784개의 원소로 이루어진 1차원 배열로 저장
#one_hot_label은 원-핫 인코딩 형태로 저장할지 정함 -> False = 숫자 형태의 레이블 저장, True = 원-핫 인코딩하여 저장
#원-핫 인코딩이란 정답을 뜻하는 원소만 1이고 나머지는 0인 배열

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)