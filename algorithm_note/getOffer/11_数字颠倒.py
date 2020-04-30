'''
@Author: your name
@Date: 2020-01-14 15:45:10
@LastEditTime : 2020-01-14 16:09:53
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\11_数字颠倒.py
'''

'''
题目描述
描述：

输入一个整数，将这个整数以字符串的形式逆序输出

程序不考虑负数的情况，若数字含有0，则逆序形式也含有0，如输入为100，则输出为001


输入描述:
输入一个int整数

输出描述:
将这个整数以字符串的形式逆序输出
'''
while True:
    try:
        num = input()
        print(num[::-1])
    except:
        break

while True:
    try:
        num = input()
        res = ''
        for i in num:
            res = i + res
        print(num[::-1])
    except:
        break


