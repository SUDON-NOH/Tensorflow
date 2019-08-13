# Linear_Regression.py
# Using tensorflow

import tensorflow as tf
tf.set_random_seed(777)     # 같이 하는 사람들이 고정된 값이서 시작하기 위해서, 속도의 차이를 없애기 위해서

# X and Y data
x_train = [1,2,3]
y_train = [1,2,3]
#y_train = [4,7,10]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = x_train * W + b        # 예측방정식

cost = tf.reduce_mean(tf.square(hypothesis - y_train ))
# 비용함수의 평균
# tf.square 제곱함수
# tf.reduce_mean 평균

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # 학습률은 직감적으로 조절해야 한다.
train = optimizer.minimize(cost)                                  # 비용을 최소화하는 w와 b를 구하라
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# start learning : 학습단계
for step in range(2001):
    sess.run(train)
    if step % 20 == 0 :     # 20번마다 줄어든 정도를 보여주도록 설정
        print(step,sess.run(cost),sess.run(W),sess.run(b))

# Test(검증) 단계


# Predict(예측) 단계
Weight = sess.run(W)
bias = sess.run(b)
print('weight = ', W,'\n' 'Bias = ', b)

x_train = [4, 5, 6]
hypothesis2 = x_train * W + b
pred = sess.run(hypothesis)
print('predict :', pred)

