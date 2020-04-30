'''
@Author: your name
@Date: 2020-04-10 09:10:41
@LastEditTime: 2020-04-10 09:54:17
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\ali02.py
'''

def S(n,m,c,sheng):
    
    sh = []
    for i in sheng:
        for j in range(i):
            sh.append()
    if n*m > sum(sheng):
        return "NO"
    else:
        for i in range(n):

            for j in range(m):
                

        return "YES"        

    



while True:
    try:
        i1 = int(input())
        for i in range(len(il)):
            i2 = input().split(" ")
            n, m, c = int(i2[0]), int(i2[1]), int(i2[2])
            sheng = []
            for i in c:
                sheng.append(int(input()))
            print(S(n,m,c,sheng))
    except:
        break


def calc(coor):
    xlis=[]
    dislis = []
    for i in range(len(coor)):
        x = coor[i][0]
        xlis.append(x)
    maxx = max(xlis)
    minx = min(xlis)
    for i in range(minx, maxx):
        port = i
        discount=0
        for j in range(len(xlis)):
            distmp = abs(xlis[j]-port)
            discount += distmp
        dislis.append(discount)
    disout = min(dislis)
    return disout

while True:
    try:
        n = int(input())
        lis = []
        for i in range(n):
            xy = input().split("")
            listmp = [int(xy[0]), int(xy[1])]
            lis.append(listmp)
        dist = calc(lis)
    except:
        break        