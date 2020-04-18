'''
@Author: your name
@Date: 2020-03-26 22:02:36
@LastEditTime: 2020-03-27 21:00:46
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\tes.py
'''


def Solu(n):
    res = []
    l = [1,1,2]
    while len(l)<=n:
        l.append(l[-1]+l[-2])
    return l[-1]
# print(Solu(8))

def calc(n,lis):
    out = [0]
    tmp = [0]
    for i in lis:
        if i ==0:
            for j in range(len(tmp)):
                tmp[j]=0
        elif i == 1:
            for j in range(len(tmp)):
                tmp[j]=tmp[j]+1
            for j in range(len(tmp)):
                out[j] = out[j] + tmp[j]
        else:
            out.append(out[-1])
            tmp1=tmp[-1]+1
            tmp.append(tmp1)
            tmp[-2] = 0
            out[-1]=out[-1]+tmp[-1]
    outout = int(sum(out)/len(out))
    return outout

while True:
    try:
        n = int(input())
        lis = []
        for i in range(n):
            ni = int(input())
            lis.append(ni)
            out = calc(n, lis)
            print(out)
    except:
        break