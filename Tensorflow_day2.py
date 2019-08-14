import tensorflow as tf
import matplotlib.pyplot as plt

# 지연실행(Lazy evaluation) 복습
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = tf.add(a , b)
sess = tf.Session()

result = sess.run(adder_node, feed_dict={a:3.2, b:4.5}) # 하나의 숫자만 들어가는 게 아니라 x 자리에 리스트 등이 들어갈 수 있다.
print('result: ',result) # 7.7


result = sess.run(adder_node, feed_dict={a:[1, 3, 5, 7],
                                         b:[2, 4, 6, 8]})
print('result: ', result) # [ 3.  7. 11. 15.]


"""

텐서플로로 예측모델 유형
1. LinearRegression
   - 입력값과 출력값이 비례관계 (선형 관계)
   - 예측방정식은 H(x) = Wx + b
   ex) 1시간 공부해서 70점, 2시간 공부해서 80점
   
2. Classification(분류) 모델

   (1) binary classification : '0과 1'처럼 답이 두 개만 존재 
       (=logistic regression)
       ex) 1시간 공부해서 합격 / 불합격
       sigmoid 함수 : 한 없이 0과 1에 가까워진다.
       
   (2) multi-nomial classification : 답이 세 가지 이상
       softmax 함수 : 최대값을 구한다.
       
"""
# 순수 python으로 LinearRegression 구현
# LinearRegression : 선형회귀
# 목표 : cost가 가장 작은 weight을 구하는 것

#   Hypothesis(가설) : 예측(가설)방정식 (H(x))

#   Cost(비용) : loss(손실), 비용함수

#   Gradient Descent Algorithm(경사하강법) : 비용함수를 미분

"""
1) 예측(가설)방정식: hx = w * x  (bias 는 0으로 가정)
    x : 입력값
    y : 답
    hx : 예측값
    
2) 비용(cost)함수

    x = [1, 2, 3]
    y = [1, 2, 3]
    # x와 y는 모두 리스트의 형태
    
def cost(x,y,w):
    c = 0
    for i in range(len(x):
        hx = w * x[i]
        loss = (hx - y[i]) ** 2         # 비용함수
        c += loss
    return c/len(x)
"""

x = [1, 2, 3]
y = [1, 2, 3]


# x와 y는 모두 리스트의 형태

def cost(x, y, w): # cost 비용의 평균
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        loss = (hx - y[i]) ** 2  # 비용함수
        c += loss
    return c / len(x)

print(cost(x, y, -1)) # 18.6666
# cost 값이 크다는 것은 -1 인 기울기는 적당한 값이 아니라고 판단 할 수 있다.
print(cost(x, y, 0)) # 4.6
print(cost(x, y, 1)) # 0.0
print(cost(x, y, 2)) # 4.6
print(cost(x, y, 3)) # 18.666

for i in range(-30, 50):
    w = i / 10
    c = cost(x, y, w)
#    print(w, c)
#    plt.plot(w, c, 'ro') # 'ro' 빨간색

plt.xlabel('wheight')
plt.ylabel('cost')
plt.title('cost function')
plt.show()



# Gradient Desent algorithm(경사하강법) : 비용함수를 미분
"""
미분 : 순간 변화량, 기울기
 - x 축으로 1만큼 이동 할 때 y 축으로 움직인 거리
    y = 3           ->    y' = 0
    y = 2 * x       ->    y' = 2    x의 제곱수를 상수와 곱하고 x의 제곱수 - 1 을 해서 미분 값을 구한다.
    y = x ** 2      ->    y' = 2x
    y = (x + 1)**2  ->    y' = 2x + 2 = 2(x + 1)
 - f(x) = a * x^n
    f'(x) = n * a * x^n-1
"""

def gradient_descent(x, y, w):
    grad = 0
    for i in range(len(x)):
        hx = w * x[i]               # w에 대한 미분
        loss = (w * x[i] - y[i])**2 # 비용함수
        """
        w를 x로 취급 나머지는 상수
        = (w * x[i])^2 - 2w*x[i]*y[i] + (y[i])^2
        = 2w(x[i])^2 - 2x[i]y[i] + 0
        = 2(w * x[i] - y[i]) * x[i]
        """
        loss_grad = 2 * (hx - y[i]) * x[i]  # 비용함수의 미분
        grad += loss_grad
    return grad/len(x)


x = [1, 2, 3]
y = [1, 2, 3]
w = 10               # 기울기는 임의의 값
old = 100
for i in range(100): # 100번 학습 시킬 것
    c = cost(x, y, w)                   # 비용이 작은 기울기를 구하자
    grad = gradient_descent(x, y, w)
    w = w - grad * 0.1  # Learning Rate : 학습률 : 0.1 : 산에서 내려올 때의 보폭
    print(i, c, w, grad) # 기울기를 미세하게 조정하는 것
    if c >= old and abs(c - old) < 1.0e-15:
        break
    old = c
    plt.plot((0, 5), (0, 5 * w))

print('weight =', w)
plt.plot(x, y, 'ro')
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.show()

