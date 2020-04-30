'''
@Author: your name
@Date: 2020-01-13 16:42:59
@LastEditTime : 2020-01-13 16:48:18
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\06_质数因子.py
'''
"""
题目描述
功能:输入一个正整数，按照从小到大的顺序输出它的所有质数的因子（如180的质数因子为2 2 3 3 5 ）

最后一个数后面也要有空格

详细描述：


函数接口说明：

public String getResult(long ulDataInput)

输入参数：

long ulDataInput：输入的正整数

返回值：

String



输入描述:
输入一个long型整数

输出描述:
按照从小到大的顺序输出它的所有质数的因子，以空格隔开。最后一个数后面也要有空格。

示例1
输入
复制
180
输出
复制
2 2 3 3 5
"""

import sys

while True:
    try:
        s = eval(input())
        while(s != 1):
            i = 2
            while True:
                if(s % i == 0):
                    print(i, end = ' ')
                    s = s / i
                    break
                i = i + 1
    except:
        break