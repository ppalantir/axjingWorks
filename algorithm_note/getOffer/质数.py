'''
@Author: your name
@Date: 2020-03-20 20:39:19
@LastEditTime: 2020-03-22 16:27:42
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\质数.py
'''

n = (i for i in range(1,101))
res=[]
for i in n:
    if i == 1:
        continue
    else:
        
        for j in range(2,i):
            if i%j == 0:
                break
        else:
            res.append(i)

print(res)

def Sol(l1, l2):
    
                

            