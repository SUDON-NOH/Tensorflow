import tensorflow as tf
# warning message 무시
import warnings
warnings.filterwarnings("ignore")

"""
pip install "numpy<1.17"
tensorflow는 numpy 1.17 버전을 아직 지원하지 않음
"""

# 상수를 변수에 저장하는 코드 : constant(상수)
hello = tf.constant('Hello, TensorFlow!')
print(hello)

"""
Tensor("Const:0", shape=(), dtype=string)
hello가 텐서플로의 텐서라는 자료형
"""

a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)
print(c)

"""
Tensor("Add:0", shape=(), dtype=int32)
그래프 = 텐서들의 연산 모음
텐서플로는 텐서와 텐서의 연산들을 먼저 정의하여 그래프를 만든다.
연산을 실행하는 코드를 넣어 '원하는 시점'에 실제 연산을 수행하도록 한다.
"""


# 그래프의 실행 : Session()와 run()을 이용
sess = tf.Session()

print(sess.run(hello))      # b'Hello, TensorFlow!'
print(sess.run([a, b, c]))  # [10, 32, 42]


# 플레이스홀더 : placeholder() : 그래프에 사용할 입력값을 나중에 받기 위해 사용하는 매개변수
# 변수 : 그래프를 최적화하는 용도로 텐서플로가 학습한 결과를 갱신하기 위해 사용하는 변수

X = tf.placeholder(tf.float32, [None, 3])
print(X)
"""
Tensor("Placeholder:0", shape=(?, 3), dtype=float32)
None 값은 크기가 정해지지 않았음을 의미
[None, 3] : Shape
두 번쨰 차원은 요소를 3개씩 가지고 있어야 한다.
"""
x_data = [[1, 2, 3], [4, 5, 6]]


# 변수 정의
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([2, 1]))
"""
W 는 [3, 2] 행렬 형태의 텐서
b 는 [2, 1] 행렬 형태의 텐서
tf.random_normal 함수를 이용해 정규분포의 무작위 값으로 초기화
다음과 같은 형태도 가능 :
 W = tf.Variable([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
"""

# 행렬의 곱셈 : tf.matmul 함수를 사용
expr = tf.matmul(X, W) + b
"""
행렬곱 A X B , 행렬 A 의 열의 수와 행렬 B의 행의 수는 같아야 한다.
행렬곱 A X B , 행렬 AB의 크기는 A의 행 개수와 B의 열 개수가 된다.
"""

sess = tf.Session()
sess.run(tf.global_variables_initializer())
"""
앞에서 정의한 변수들을 초기화하는 함수
기존에 학습한 값들을 가져와서 사용하는 것이 아닌 처음 실행하는 것이라면,
연산을 실행하기 전에 반드시 이 함수를 이용해 변수들을 초기화해야 한다.
"""

print("=========== x_data ===========")
print(x_data)
print("===========  W ===============")
print(sess.run(W))
print("===========  b ===============")
print(sess.run(b))
print("=========== expr =============")
print(sess.run(expr, feed_dict={X: x_data}))

sess.close()
"""
feed_dict 매개변수는 그래프를 실행할 때 사용할 입력값을 지정
=========== x_data ===========
[[1, 2, 3], [4, 5, 6]]
===========  W ===============
[[ 2.5381782   0.25653988]
 [-1.1872144   0.36564836]
 [ 0.5912495   0.07532614]]
===========  b ===============
[[ 0.2410574]
 [-1.9882145]]
=========== expr =============
[[2.1785555 1.4548724]
 [5.7759237 1.3181438]]
"""

y_data = [1, 2, 3]
x_data = [1, 2, 3]

# X와 Y의 상관관계를 설명하기 위한 변수들인 W와 b를 각각 균등분포를 가진 무작위 값으로 초기화
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
"""
W와 b를 각각 -1.0에서부터 1.0 사이의 균등분포를 가진 무작위 값으로 초기화한다.
"""

# Placeholder를 설정
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# X, Y의 상관 관계를 분석하기 위한 수식 작성
hypothesis = W * X + b
"""
X가 주어졌을 때 Y를 만들어 낼 수 있는 W(가중치:weight)와 b(편향:bias)를 찾아내겠다는 의미
"""

# 손실 함수 작성
cost = tf.reduce_mean(tf.square(hypothesis - Y))
"""
손실함수(loss function)는 한 쌍(x, y)의 데이터에 대한 손실값을 계산하는 함수
손실값: 실제값과 모델로 예측한 값이 얼마나 차이가 나는가를 나타내는 값
손실값이 작을수록 그 모델이 X와 Y의 관계를 잘 설명하고 있다는 뜻
이 손실을 전체 데이터에 대해 구한 경우 비용(cost)이라고 한다
"""

# 경사하강법(gradient descent) 최적화 함수를 이용해 손실값을 최소화하는 연산 그래프를 생성
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)
"""
최적화 함수: 가중치와 편향 값을 변경해가면서 손실값을 최소화하는 가장 최적화된 가중치와 편향값을 찾아주는 함수
경사하강법: 함ㅈ수의 기울기를 구하고 기울기가 낮은 쪽으로 계속 이동시키면서 최적의 값을 찾아나가는 방법
learing_rate(학습률)은 학습을 얼마나 '급하게' 할 것인가를 설정하는 값
값이 너무 크면 최적의 손실값을 찾지 못하고 지나치게 되고, 값이 너무 작으면 학습 속도가 매우 느려진다.
하이퍼파라미터(hyperparameter) : 학습을 진행하는 과정에 영향을 주는 변수
"""

# 결과 확인
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # with 기능을 이용해 세션블록을 만들고 세션 종료를 자동으로 처리하도록
    for step in range(100):             # 학습은 100번 수행
        _, cost_val = sess.run([train_op, cost],
                               feed_dict = {X: x_data, Y: y_data})  # feed_dict를 통해 상관관계를 알아낸다.
        print(step, cost_val, sess.run(W), sess.run(b))

    print("\n==============Test=================")

    print("X : 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
    print("X : 2.5, Y:", sess.run(hypothesis, feed_dict={X: 5}))

