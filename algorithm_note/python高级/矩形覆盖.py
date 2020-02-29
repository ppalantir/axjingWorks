'''
@Author: your name
@Date: 2020-02-20 15:01:13
@LastEditTime: 2020-02-20 15:01:37
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\python高级\矩形覆盖.py
'''

'''
我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

比如n=3时，2*3的矩形块有3种覆盖方法：
'''

class Solution:
    def rectCover(self, number):
        # write code here

        res = [0,1,2]
        while len(res) <= number:
            res.append(res[-1] + res[-2])
        return res[number]
