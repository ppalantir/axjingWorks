'''
@Author: your name
@Date: 2020-02-20 14:26:54
@LastEditTime: 2020-02-20 14:27:10
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\python高级\斐波那契数列.py
'''
'''
大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。
n<=39
'''

class Solution:
    def Fibonacci(self, n):
        # write code here
        list = [0, 1]
        while n > 0:
            list[0], list[1] = list[1], list[0]+list[1]
            n -= 1
        return list[0]