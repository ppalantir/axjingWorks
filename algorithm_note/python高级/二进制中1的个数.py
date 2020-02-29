'''
@Author: your name
@Date: 2020-02-20 16:23:02
@LastEditTime: 2020-02-20 16:23:37
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\python高级\二进制中1的个数.py
'''

'''
输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
'''

# -*- coding:utf-8 -*-
class Solution:
        # write code here
    def NumberOf1(self, n):
        count = 0
        while n&0xffffffff != 0:
            count += 1
            n = n & (n-1)
        return count