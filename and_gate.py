# and_gate.py

def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = w1 * x1 + w2 * x2
    if tmp <= theta : # 임계값 : 한계값
        return 0
    elif tmp > theta:
        return 1

print(AND(0, 0)) # 0 False
print(AND(0, 1)) # 0 False
print(AND(1, 0)) # 0 False
print(AND(1, 1)) # 1 True



