# binary classification : 0과 1 처럼 답이 두개만 존재
# ( = logistic regression)
# 예측 방정식으로 sigmoid 함수 사용
# sigmoid 보다 ReLu가 성능이 더 좋다고 알려짐 (ReLu : x<0 면 y = 0, x>0 면 y = 1 로 바꾸는 방법)

"""
합격 or 불합격과 같은 이분법을 나타내는데 사용
1 또는 0에 수렴한다
0.5를 기준으로 0.5 이상이면 1, 이하면 0
"""


# 데이터를 70%(train), 30%(test)

# logistic Regression diabetes
import tensorflow as tf
import numpy as np

# ===================================================================================================================

# 전체데이터
xy = np.loadtxt('data-03-diabetes.csv',delimiter = ',', dtype = np.float32)
print(xy.shape) # (759, 9)


# ===================================================================================================================

# 학습데이터(train data set , 70% : 531개)
# Overfit - ting 과적합 : 이유: x의 값이 여러 개일 경우에 자주 나타남
x_train = xy[0:531, 0:-1] # 답은 제외하고 해야하기 때문에 범위를 -1로 해준다
print(x_train.shape) # (531, 8)
y_train = xy[0:531, [-1]] # 그냥 -1만 넣었을 경우 1차원 array로 나오기 때문에 괄호를 취해준다.
print(y_train.shape) # (531, 1)


# ===================================================================================================================

# 검증데이터(test data set, 30% : 228개)
x_test = xy[531:, 0:-1]
print(x_test.shape) # (228, 8)
y_test = xy[531:, [-1]]
print(y_test.shape) # (228, 8)


# ===================================================================================================================

# Build Graph
X = tf.placeholder(tf.float32, shape = [None, 8])
Y = tf.placeholder(tf.float32, shape = [None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# 예측 방정식(Hypothesis)
# hypothesis = tf.div(1., 1. + tf.exp(tf.matmul(X,W) + b)) # sigmoid 사용 전 식
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# 비용 함수 (logistic regression)
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1 - Y)*tf.log(1 - hypothesis))

# SGD Optimizer = 경사하강법
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01)
train = optimizer.minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# 학습단계
for step in range(100001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train], feed_dict={X:x_train, Y:y_train})

    if step % 10000 == 0:
        print(step, cost_val, W_val, b_val)


# ===================================================================================================================

# 정확도 측정: Accuracy Report
# 검증 데이터(test data set, 30% : 228개)

# 정확도 계산(accuracy computation)
predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict , Y), dtype = tf.float32))
# tf.equal : 괄호 안에 있는 두가지가 같으면 True 다르면 False
# tf.cast : True 면 1 False 면 0 으로 바꿔준다.
# tf.reduce_mean : 다 더한 후 평균을 계산

h,p,a = sess.run([hypothesis, predict, accuracy],
                 feed_dict= {X:x_test , Y:y_test})

print('\nHypothesis:', h, '\nPredict:', p, '\nAccuracy:', a) # Accuracy: 0.7763158

# 예측 단계(Predict)
print(sess.run(predict, feed_dict={X:x_test}))