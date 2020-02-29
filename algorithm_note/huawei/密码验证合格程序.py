'''
@Author: your name
@Date: 2020-02-23 14:14:08
@LastEditTime: 2020-02-23 14:21:33
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\密码验证合格程序.py
'''
'''
题目描述
密码要求:
1.长度超过8位
2.包括大小写字母.数字.其它符号,以上四种至少三种
3.不能有相同长度超2的子串重复

说明:长度超过2的子串
输入描述:
一组或多组长度超过2的子符串。每组占一行

输出描述:
如果符合要求输出：OK，否则输出NG

示例1
输入
复制
021Abc9000
021Abc9Abc1
021ABC9000
021$bc9000
输出
复制
OK
NG
NG
OK
'''

import sys
import re

for line in sys.stdin:

    line = line.strip()
    #1
    if len(line) <= 8:
        print("NG")
        continue
    #2
    count = 0
    if re.search('[0-9]',line): count += 1
    if re.search('[a-z]',line): count += 1
    if re.search('[A-Z]',line): count += 1
    if re.search('[^a-zA-Z0-9]',line): count += 1
    if count < 3: 
        print("NG")
        continue
    #3:
    if re.search(r'.*(...)(.*\1)', line):
        print("NG")
        continue
     
    print("OK")