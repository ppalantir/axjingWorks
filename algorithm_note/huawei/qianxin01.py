'''
@Author: your name
@Date: 2020-02-28 18:33:45
@LastEditTime: 2020-02-28 19:58:59
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\qianxin01.py
'''



    

yingbi = 10000
jinbi = 10
while True:
    try:
        
        n = int(input())
        if n > 0 and n<1001:
            p = 0
            for i  in range(n):
                p = p + (jinbi-i)/(yingbi-i)
        
        
    except:
        break

def digui(n):
    raw = 1
    zi = 0
    if n % 5 == 0:
        



while True:
    try:
        
        n = int(input())
        if n > 0 and n<101:
            res = 1
            zi = 0
            for i  in range(1, n+1):
                if i % 5 == 0:
                    a = i // 5
                    zi = zi + 2**a
                    res = res + zi + 1
                else:
                    res = res + zi + 1
        print(res)
        
    except:
        break

while True:
    try:
        
        n = int(input())
        if n > 0 and n<101:
            res = 0
            zi = 0
            for i  in range(1, n+1):
                if i+1 % 5 == 0:
                    res = res + 1 + zi + 1
                else:
                    res = res + 1
        print(res)
        
    except:
        break

def calc(n):
    lis = [1]
    ite = int((n-1)/4)
    for i in range(ite):
        lis.append(2**i)
    return sum(lis)

while True:
    try:
        n = int(input())
        print(calc(n))
    except:
        break

def jiecheng(n):
    if n==1:
        return 1
    else:
        return n*jiecheng(n-1)

def calcC(a,b):
    fz = jiecheng(b)
    fm = jiecheng(b-a)*jiecheng(a)
    return fz/fm
def gailv(n):
    if n>990:
        return ("%.6f"%1)
    elif n == 0:
        return ("%.6f"%0)
    else:
        lis = [0.01]
        for i in range(n-1):
            tmp = calcC(1, 1000-i)*lis[-1]
            lis.append(tmp)
    return ("%.6f"%lis[-1])

while True:
    try:
        n = int(input())
        print(gailv(n))
    except:
        break
        