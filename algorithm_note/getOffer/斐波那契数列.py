'''
@Author: your name
@Date: 2020-04-07 09:36:35
@LastEditTime: 2020-04-07 09:42:18
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\斐波那契数列.py
'''

class Solution():
    def Fibonacci(self,n):
        res = [0, 1, 1]
        while len(res) <= n:
            res.append(res[-1]+res[-2])
        return res[n]