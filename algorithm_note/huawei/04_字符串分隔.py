'''
@Author: your name
@Date: 2020-01-12 20:22:51
@LastEditTime : 2020-01-13 16:27:29
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\04_字符串分隔.py
'''
"""
题目描述
•连续输入字符串，请按长度为8拆分每个字符串后输出到新的字符串数组；
•长度不是8整数倍的字符串请在后面补数字0，空字符串不处理。
输入描述:
连续输入字符串(输入2次,每个字符串长度小于100)

输出描述:
输出到长度为8的新字符串数组
"""
while True:
    try:
        a = int(input())
        for i in range(a):
            s = input()
            while len(s) >= 8:
                print(s[:8])
                s = s[8:]

            print(s.ljust(8, "0"))  # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串。如果指定的长度小于原字符串的长度则返回原字符串。
    except:
        break

import math
def calc(string):
    num=math.ceil(len(string)/8)
    modd=len(string)/8
    out=''
    for i in range(num):
        for j in range(8):
            if i*8+j<len(string):
                out+=string[i*8+j]
            else:
                out+='0'
        print(out)
        out=''
while True:
    try:
        lis1=input()
        lis2=input()
        calc(lis1)
        calc(lis2)
    except:
        break

