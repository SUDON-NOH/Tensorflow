# Tensorflow_변수

a = 10
b = [1, 2, 3, 4, 5]
c = (a, b)
a = a + 1
print(a, b, c)

import tensorflow as tf

# Variable vs. placeholder : tensorflow 전용 변수

# [1] Variable : tensorflow 내부에서 연산할 때 사용되는 변수
#                weight 과 bias
# 머신러닝의 목표 : 주어진 빅데이터를 통해서 앞으로 어떤 데이터가 나왔을 때 그 결과를 예측
# h(x) = Wx + b
"""
weight : 가중치
weight을 지나치게 작게 주면 1에 가까워지기 어려워 값들의 상관관계를 구분하기 어렵다.
h(x) 값에 영향을 주는 정도
현실은 h(x) = (Wx + b) + (Wx + b) ...
bias 값이 크면 아무리 Wx의 값이 작아도 1000이기 때문에 구분하기 어렵다.
"""

"""
활성화
bias = 0.5 라면
0.7 -> 1
0.9 -> 1
0.3 -> 0
"""

var1 = tf.Variable(10) # node
print(var1)
# <tf.Variable 'Variable:0' shape=() dtype=int32_ref>

var2 = tf.Variable(20)
print(var2)
# <tf.Variable 'Variable_1:0' shape=() dtype=int32_ref>

var3 = var1 * var2 # 하나의 그래프를 할당한 것
print(var3)
# Tensor("mul:0", shape=(), dtype=int32)

# 지연실행(Lazy evaluation)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
var_result = sess.run(var1)
print('var1 = ', var_result)

var2_result = sess.run(var2)
print('var2 = ', var2_result)

print('var3 = ', sess.run(var3))

# node : 연산  (커피 머신)
# edge : 데이터 (커피 알갱이)

# [2] placeholder 변수 : 초기값이 정해지지 않고
# sess.run() 입력시 입력
# feed_dict{X:, Y:}
# 입력변수(빅데이터들) : X(feature), Y(답)
holder1 = tf.placeholder(tf.float32, shape = [1])
result1 = holder1 * 2 # 예측 공식
print(result1) # Tensor("mul_1:0", shape=(1,), dtype=float32)
result2 = sess.run(result1, feed_dict = {holder1 : [10]}) # feed_dict 는 node에 들어갈 값들을 정의해주는 기능을 한다.
print(result2) # [20.]

w