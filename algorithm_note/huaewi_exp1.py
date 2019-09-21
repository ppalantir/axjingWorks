import os
import sys
import numpy as np

def input_data():
    A = input("请输入序列A:").strip("\n")
    B = input("请输入序列B:").strip()
    R = int(input("请输入距离R:").strip())
    A_l = A.split(",")
    B_l = B.split(",")
    for i in range(len(A_l)):
        A_l[i] = int(A_l[i])
    for i in range(len(B_l)):
        B_l[i] = int(B_l[i])
    
    if len(A_l) <= 50 and len(B_l) <= 50 and max(A_l)<=65535 and max(B_l)<=65535:
        for i in range(len(A_l)):
            for j in range(len(B_l)):
                if A_l[i]<=B_l[j] and abs(A_l[i] - B_l[j]) <= R:
                    return (A_l[i], B_l[j])
            


import re

def calc(a, b,r):
    out = []
    if len(a)>len(b):
        for i in range(len(a)):
            b.append(b[-1])
    for i in range(len(a)):
        for j in range(len(b)):
            tmp = [a[i], b[j]]
            if tmp[0] <= tmp[1] and abs(tmp[0] - tmp[1]) <= r:
                out.append(tmp)
    return out

while True:
    try:
        abr = input()
        a = re.findall(".*A={(.*)},B=.*", abr)
        b = re.findall(".*B={(.*)},R=.*", abr)
        r = re.findall(".*R=(.*)", abr)

        a = a[0].split(",")
        b = b[0].split(",")
        for i in range(len(a)):
            a[i] = int(a[i])
        for i in range(len(b)):
            b[i] = int(b[i])

        r = int(r[0])
        print(calc(a, b, r))
    except:
        break


if __name__ == "__main__":
   
    print(input_data())