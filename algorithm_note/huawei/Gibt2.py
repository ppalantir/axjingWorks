'''
@Author: your name
@Date: 2020-03-14 17:37:36
@LastEditTime: 2020-03-14 20:36:27
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\Gibt2.py
'''
from itertools import permutations

def calc(n):
    outlis=[]
    #book=[]
    inlis=['A','A','B','B','C','C','D','D']
    book=list(permutations(inlis,n))
    for i in range(len(book)):
        acc=0
        for j in range(len(book[i])):        
            if book[i][j]=='A' or book[i][j]=='C':
                acc+=1
        if acc%2==0 and book[i] not in outlis:
            outlis.append(book[i])
    print(outlis)
    return len(outlis)%1000000007

'''while True:
    try:
        st_in=input()
        st_out=calc(st_in)
        print(st_out)
    except:
        break'''
o=calc(3)
print(o)


from intertools import permutations, combinations
def calc(lis):
    alllis=[]
    for i in lis:
        alllis+=i
    allcom = list(combinations(alllis,4))
    return len(allcom)

while True:
    try:
        nm=input().split('')
        n = int(nm[0])
        m = int(nm[1])
        inlis=[]
        for i in range(n):
            tmplis = []
            tmp=input()
            for j in range(m):
                tmplis.append(tmp[j])
            inlis.append(tmplis)
        out=calc(inlis)
    except:
        break