'''
@Author: your name
@Date: 2020-01-12 19:45:22
@LastEditTime : 2020-01-12 19:46:04
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\最后一个字符串长度.py
'''
import sys
'''
题目描述
计算字符串最后一个单词的长度，单词以空格隔开。
输入描述:
一行字符串，非空，长度小于5000。

输出描述:
整数N，最后一个单词的长度。
'''
s = sys.stdin.readline()
l = len(s.split()[-1:][0])
print(l)