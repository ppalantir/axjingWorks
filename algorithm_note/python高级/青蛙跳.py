'''
@Author: your name
@Date: 2020-02-20 14:53:08
@LastEditTime: 2020-02-20 14:54:04
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\python高级\青蛙跳.py
'''
''''
一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
'''

class Solution:
    def jumpFloorII(self, number):
        # write code here
        if number <= 0:
            return 0
        else:
            return 2**(number-1)