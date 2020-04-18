'''
@Author: your name
@Date: 2020-03-14 17:02:45
@LastEditTime: 2020-03-14 17:29:15
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\Gbit.py
'''
def calc(st):
    out = ''
    book=[]
    for i in range(len(st)):
        tmp=ord(st[i])
        if tmp>=97:
            tmp=tmp-32

        if tmp not in book:
            out = out+st[i]
            book.append(tmp)
    return out
    
while True:
    try:
        st_in = input()
        st_out = calc(st_in)
        print(st_out)

    except:
        break

def solution(n):
    l = ["A", "B", "C", "D"]
    st = ''

def qp(x, num):
    mod = 100
    res = 1
    while num>0:
        if num&1 ==1:
            res = (res*x)%mod
        x = (x*x)%mod
        num+=1
    return res




while True:
    n = int(input())
    print(8)
    
