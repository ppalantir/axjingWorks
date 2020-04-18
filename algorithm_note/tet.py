'''
@Author: your name
@Date: 2020-03-27 08:28:33
@LastEditTime: 2020-03-27 09:14:57
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\tet.py
'''
import random
def gongj():
    a = 100 #敌人
    b = 100
    g = [1,2,3,4,5]
    while a == 0 and b>0:
        for j in range(len(g)):
            m1 = random.choice(g)
            # print(m1)
            a = a-g[j]
            b= b-m1
            print(g[j])

gongj()