# cnn_basic.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
# filter는 사실상 weight 이라고 볼 수 있다.
#(1, 3, 3, 1) 3*3 Greyscale toy image 4차원 ( 0번 축, 1번 축, 2번 축, 3번 축)
# 1: 이미지의 개수 : None으로 변경 가능
# 3: 이미지의 가로 픽셀 수
# 3: 이미지의 세로 픽셀 수
# 1: color 수 , 1은 Grey
image = np.array([[[[1],[2],[3]],
                   [[4],[5],[6]],
                   [[7],[8],[9]]]], dtype = np.float32)


# print(image)
"""
[[[[1.]
   [2.]
   [3.]]
   
  [[4.]
   [5.]
   [6.]]
   
  [[7.]
   [8.]
   [9.]]]]
"""
# print(image.shape) # (1, 3, 3, 1)
plt.imshow(image.reshape(3,3), cmap = 'Greys') # 9개의 픽셀을 확대해서 봄
plt.show()



# sess = tf.InteractiveSession()
# a = tf.constant(10)
# b = tf.constant(20)
# add_node = tf.add(a, b)
# print(add_node.eval()) # session.run을 안쓰고 바로 사용할 수 있다
# sess.close()


# conv2d, Padding
# image : (1, 3, 3, 1), filter : (2, 2, 1, 1,), stride:(1, 1)
"""
filter : (2, 2, 1, 1)
2: 가로의 수
2: 세로의 수
1: colot 수
1: filter의 수 1: 1장의 이미지만 출력

(N - F)/stride + 1 공식을 사용해서 출력이미지의 shape을 알 수 있다.
(3 - 2)/1 + 1 = 2 # Zero Padding을 하지 않은 경우

(4 - 2)/1 + 1 = 3 # Zero Padding을 한 경우
"""


# padding을 안했을 때 : VALID
# 출력이미지 : (1, 2, 2, 1)
# filter:(2, 2, 1, 1)
weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]]) # 보통은 random 값으로 한다.

conv2d = tf.nn.conv2d(image, weight, strides = [1, 1, 1, 1],
                     padding = 'VALID') # 이미지와 filter가 곱해진다.

conv2d_img = conv2d.eval()  # (1, 2, 2, 1)
print(conv2d_img)
"""
[[[[12.]
   [16.]]
  [[24.]
   [28.]]]]
"""
print('conv2d_img.shape:', conv2d_img.shape)

# 시각화
for i, one_image in enumerate(conv2d_img): # 어떤 데이터에서 index와 데이터를 그대로 뽑아낸다.
    print(one_image.reshape(2, 2))
    plt.imshow(conv2d_img.reshape(2, 2), cmap='Greys')
"""
[[12. 16.]
 [24. 28.]]
"""
plt.show()




# =================================================================================================================

# padding을 했을 때 : SAME
# image : (1, 3, 3, 1)
# filter:(2, 2, 1, 1)
# 출력이미지 : (1, 3, 3, 1)

weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]]) # 보통은 random 값으로 한다.

conv2d = tf.nn.conv2d(image, weight, strides = [1, 1, 1, 1],
                     padding = 'SAME') # 이미지와 filter가 곱해진다.

conv2d_img = conv2d.eval()  # (1, 3, 3, 1)
print(conv2d_img)
"""
[[[[12.]
   [16.]
   [ 9.]]
  [[24.]
   [28.]
   [15.]]
  [[15.]
   [17.]
   [ 9.]]]]
"""
print('conv2d_img.shape:', conv2d_img.shape)

# conv2d : (1, 3, 3, 1)     conv2d 를 거치면 0번과 3번의 축이 바뀌어서 출력됨
# 1 : color 수
# (N - F)/stride + 1
# ((3 + 1) - 2)/1 + 1 = 3
# ((3 + 1) - 2)/1 + 1 = 3
# 1 : 이미지의 개수 (사용된 필터의 개수)

# 시각화
for i, one_image in enumerate(conv2d_img): # 어떤 데이터에서 index와 데이터를 그대로 뽑아낸다.
    print(one_image.reshape(3, 3))
    plt.imshow(one_image.reshape(3, 3), cmap='Greys')
"""
[[12. 16.  9.]
 [24. 28. 15.]
 [15. 17.  9.]]
""" # feature 추출
plt.show()

# =================================================================================================================

# conv2d : 3 filters, Padding : SAME (zero padding 함)
# image : (1, 3, 3, 1)
# filter : (2, 2, 1, 3) # 필터를 3번 사용
# 출력이미지 : (1, 3, 3, 3)
# 1 : color 수
# (N - F)/stride + 1
# ((3 + 1) - 2)/1 + 1 = 3
# ((3 + 1) - 2)/1 + 1 = 3
# 3 : 이미지의 개수 (사용된 필터의 개수)

weight = tf.constant([[[[1., 10., -1]],[[1., 10., -1]]],
                      [[[1., 10., -1]],[[1., 10., -1]]]])
# 1 1 1 1 1장
# 10 10 10 10 1장
# -1 -1 -1 -1 1장

conv2d = tf.nn.conv2d(image, weight, strides = [1, 1, 1, 1],
                     padding = 'SAME')
conv2d_img = conv2d.eval()
print(conv2d_img.shape)# (1, 3, 3, 3)
print(conv2d_img)
"""
[[[[ 12. 120. -12.]
   [ 16. 160. -16.]
   [  9.  90.  -9.]]
   
  [[ 24. 240. -24.]
   [ 28. 280. -28.]
   [ 15. 150. -15.]]
   
  [[ 15. 150. -15.]
   [ 17. 170. -17.]
   [  9.  90.  -9.]]]]
"""
conv2d_img = np.swapaxes(conv2d_img, 0, 3) # conv2d_img의 0번과 3번의 축을 바꾼다.
print(conv2d_img.shape) # (3, 3, 3, 1)

# 시각화
for i, one_image in enumerate(conv2d_img): # 어떤 데이터에서 index와 데이터를 그대로 뽑아낸다.
    print(one_image.reshape(3, 3))
    plt.subplot(1, 3, i + 1) # 1행 3열의 형식으로 3개의 이미지를 순차적으로 출력
    plt.imshow(one_image.reshape(3, 3), cmap='Greys')

plt.show()

# =================================================================================================================

# max pooling : (1, 2, 2, 1) -> (1, 2, 2, 1) , Padding = SAME
image = np.array([[[[4],[3]],
                   [[2],[1]]]], dtype = np.float32)
print(image.shape)
# (1, 2, 2, 1)

plt.imshow(image.reshape(2, 2), cmap = 'Greys')
plt.show()

pool = tf.nn.max_pool(image, ksize = [1, 2, 2, 1], strides = [1, 1, 1, 1],
                      padding = 'SAME') # ksize 는 filter size와 비슷하게 생각해도 된다.
print(pool.shape)   # (1, 2, 2, 1)
print(pool.eval())
"""
[[[[4.]
   [3.]]

  [[2.]
   [1.]]]]
"""

# =================================================================================================================

# MNIST image loading
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

image = mnist.train.images[0].reshape(28,28)
plt.imshow(image, cmap='gray')
plt.show()

# =================================================================================================================

# MNIST Convolution layer
# image : (1, 28, 28, 1)
img = image.reshape(-1,28,28,1) # 자동 배정
print(img.shape)


W = tf.Variable(tf.random_normal([3, 3, 1, 5]), name = 'weight')

conv2d = tf.nn.conv2d(img,W,strides=[1,2,2,1],
                      padding = 'SAME')
print(conv2d) # shape = (1, 14, 14, 5)
# 1 : color 'gray'
# 5 : filter 5장
# (28 - 3)/2 + 1 = 14

sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
print(conv2d_img.shape) # (1, 14, 14, 5)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
print(conv2d_img.shape) # (5, 14, 14, 1)


# 시각화
for i, one_image in enumerate(conv2d_img): # 어떤 데이터에서 index와 데이터를 그대로 뽑아낸다.
    print(one_image.reshape(14, 14))
    plt.subplot(1, 5, i + 1) # 1행 3열의 형식으로 3개의 이미지를 순차적으로 출력
    plt.imshow(one_image.reshape(14, 14), cmap='Greys')

plt.show()

# ================================================================================================================

# MNIST Max pooling
# conv2d : (1, 14, 14, 5), kernel size:(2, 2), strides:(2, 2)
# padding: 'SAME'
# Output size : (1, 7, 7, 5)
# (N-F)/stride + 1
# (14 - 2)/2 + 1 = 7

sess.run(tf.global_variables_initializer())
pool = tf.nn.max_pool(conv2d, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
                      padding = 'SAME')

pool_img = pool.eval()
print(pool_img.shape)   # (1, 7, 7, 5)
pool_img = np.swapaxes(pool_img, 0, 3)
print(pool_img.shape)   # (5, 7, 7, 1)

# 시각화
for i, one_image in enumerate(pool_img):
    plt.subplot(1, 5, i + 1)
    plt.imshow(one_image.reshape(7, 7), cmap='Greys')

plt.show()
