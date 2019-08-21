import tensorflow as tf
import matplotlib.pyplot as plt
 
# X, Y를 placeholder 를 사용하여 구현

W = tf.Variable(tf.random_normal([1]), name = 'weight') # 1차원
b = tf.Variable(tf.random_normal([1]), name = 'bias')

X = tf.placeholder(tf.float32, shape = [None]) # None은 미지수이다.
Y = tf.placeholder(tf.float32, shape = [None])


hypothesis = X * W + b        # 예측방정식

cost = tf.reduce_mean(tf.square(hypothesis - Y))
# 비용함수의 평균
# tf.square 제곱함수
# tf.reduce_mean 평균

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # 학습률은 직감적으로 조절해야 한다.
train = optimizer.minimize(cost)                                  # 비용을 최소화하는 w와 b를 구하라
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # Variable 값을 초기화하는 작업

# start learning : 학습단계
for step in range(2001):
    cost_val, W_val, b_val, _ =\
        sess.run([cost, W, b, train],  # cost W b train 모두 동시에 run # return 값이 4개가 나온다
             # feed_dict = {X:[1, 2, 3], Y:[1, 2, 3]})
             # feed_dict = {X:[1, 2, 3], Y:[2, 4, 6]})
             feed_dict = {X:[1, 2, 3], Y:[2, 4, 7]})
    if step % 20 == 0 :     # 20번마다 줄어든 정도를 보여주도록 설정
        print(step, cost_val, W_val, b_val)
"""
for문을 돌릴수록
cost 값이 0에 가까워지지만 한없이 돌려도 그 값들의 차이가 많이 나지는 않는 경우가 있다.
그럴 때 멈춰야 하는데, 그때 break를 사용한다.
마지막에 나온 W_val(즉, 기울기 값)와 b_val(bias 값)를 이용해 예측단계에서 예측한다.
예측 단계에서 나오는 결과 값은 feed_dict로 부여한 X들의 값, 마지막 W_val, b_val를 이용한다.
이 값들은 내가 가설로 설정한 함수에 입력되어 계산한 예측 값이다.
"""


# Test(검증) 단계


# Predict(예측) 단계
print(sess.run(hypothesis, feed_dict = {X:[4, 5, 6]})) # X값을 설정해서 예측
print(sess.run(hypothesis, feed_dict = {X:[3.7]})) # X값을 설정해서 예측



#######################################################################################################

# Car 의 속도(X)와 제동거리(Y)
# 데이터셋 가져오기
import numpy as np
xy = np.loadtxt('cars.csv',unpack=True, delimiter=',',skiprows=1)
x = xy[0]       # 속도 # 0번 인덱스
y = xy[1]       # 거리 # 1번 인덱스
"""
unpack = True : 세로 column 으로 있었던 데이터셋을 가로 row로 만들어준다.
                행과 열을 Transpose하여 읽어온다.
delimiter : txt file의 구분자로 밝혀준다.
skiprows : 첫 행 하나를 skip하고 읽어온다. header를 제거할 경우 사용
"""
#######################################################################################################
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

X = tf.placeholder(tf.float32, shape = [None]) # None은 미지수이다.
Y = tf.placeholder(tf.float32, shape = [None])


hypothesis = X * W + b        # 예측방정식

cost = tf.reduce_mean(tf.square(hypothesis - Y))
# 비용함수의 평균
# tf.square 제곱함수
# tf.reduce_mean 평균

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) # 학습률은 직감적으로 조절해야 한다.
train = optimizer.minimize(cost)                                  # 비용을 최소화하는 w와 b를 구하라
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

"""
학습률을 0.01로 했을 때
오버슈팅된다.
0 2443.8179 [15.246003] [0.7707913]
20 5.1873024e+28 [6.0135213e+13] [3.5018286e+12]
40 inf [2.9304315e+26] [1.706466e+25]
60 nan [nan] [nan]
80 nan [nan] [nan]
100 nan [nan] [nan]
120 nan [nan] [nan]
140 nan [nan] [nan]
160 nan [nan] [nan]
...
"""

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# start learning : 학습단계
for step in range(40001):
    cost_val, W_val, b_val, _ =\
        sess.run([cost, W, b, train],  # cost W b train 모두 동시에 run # return 값이 4개가 나온다
             feed_dict = {X:x, Y:y})
    if step % 20 == 0 :     # 20번마다 줄어든 정도를 보여주도록 설정
        print(step, cost_val, W_val, b_val)

"""
for문을 돌릴수록
cost 값이 0에 가까워지지만 한없이 돌려도 그 값들의 차이가 많이 나지는 않는 경우가 있다.
그럴 때 멈춰야 하는데, 그때 break를 사용한다.
마지막에 나온 W_val(즉, 기울기 값)와 b_val(bias 값)를 이용해 예측단계에서 예측한다.
예측 단계에서 나오는 결과 값은 feed_dict로 부여한 X들의 값, 마지막 W_val, b_val를 이용한다.
이 값들은 내가 가설로 설정한 함수에 입력되어 계산한 예측 값이다.
"""


# Test(검증) 단계


# Predict(예측) 단계
print(sess.run(hypothesis, feed_dict = {X:[25]})) # X값을 설정해서 예측
print(sess.run(hypothesis, feed_dict = {X:[30]})) # X값을 설정해서 예측
print(sess.run(hypothesis, feed_dict = {X:[25, 67, 45]})) # X값을 설정해서 예측

# 시각화 : matplotlib사용
def prediction(x,W,b):
    return W*x + b

plt.plot(x,y,'ro')
plt.plot((0,25),(0,prediction(25,W_val,b_val)))
plt.plot((0,25),(prediction(0,W_val,b_val),prediction(25,W_val,b_val)))
plt.show()

#######################################################################################################
# multi-variable linear regression
# x 값(입력 변수)이 여러 개(2개 이상)인 경우
"""
# 행이 3개(x1, x2, x3)인 행렬을 계산해야 한다.

H(x1, x2, x3) = w1 * x1 + w2 * x2 + w3 * x3 + b
   X         W        Y
(N * 3) * (3 * 1) = (N, 1)
"""


# 입력데이터 준비

x1 = [73., 93., 89., 96., 73.]
x2 = [80., 88., 91., 98., 66.]
x3 = [75., 93., 90., 100., 70.]

y = [152., 185., 180., 196., 142.] # 점을 찍는 건 float 이라는 뜻

X1 = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.float32)
X3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

# Build Graph
W1 = tf.Variable(tf.random_normal([1]), name= 'weight1')
W2 = tf.Variable(tf.random_normal([1]), name= 'weight2')
W3 = tf.Variable(tf.random_normal([1]), name= 'weight3')

b = tf.Variable(tf.random_normal([1]), name = 'bias')

    # 예측 방정식
hypothesis = X1 * W1 + X2 * W2 + X3 * W3 + b

    # 비용 함수
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5) # 10의 -5승
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습단계
for step in range(100001):

    cost_val, W1_val , W2_val, W3_val, b_val, _ = \
        sess.run([cost, W1, W2, W3, b, train],
                 feed_dict={X1:x1, X2:x2, X3:x3, Y: y})
    if step % 20 == 0:
        print(step, cost_val, W1_val, W2_val, W3_val, b_val)

# predict(예측)
x1 = [73., 93., 89., 96., 73.]
x2 = [80., 88., 91., 98., 66.]
x3 = [75., 93., 90., 100., 70.]

print(sess.run(hypothesis, feed_dict={X1:x1, X2:x2, X3:x3}))

#######################################################################################################

# matrix  사용: tf.matmul() 사용
# x_data : [5,3]
x_data = [[73.,80.,75.],
          [93.,88.,93.],
          [89.,91.,90.],
          [96.,98.,100.],
          [73.,66.,70.]]
# y_data : [5,1]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

X = tf.placeholder(tf.float32,shape=[None,3])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
# 2차원
# 3 : X의 열
# 1 : Y의 열
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5) # 10의 -5승
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# start training
for step in range(100001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
                 feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)


# predict : test model
x_data = [[73.,80.,75.],
          [93.,88.,93.],
          [89.,91.,90.],
          [96.,98.,100.],
          [73.,66.,70.]]
print(sess.run(hypothesis, feed_dict = {X:x_data}))
#  [[151.44507]
#  [184.6806 ]
#  [180.83571]
#  [196.01993]
#  [141.9165 ]]
