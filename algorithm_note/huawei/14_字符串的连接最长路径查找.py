'''
@Author: your name
@Date: 2020-01-14 16:29:14
@LastEditTime : 2020-01-14 16:33:10
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\14_字符串的连接最长路径查找.py
'''

'''
题目描述
给定n个字符串，请对n个字符串按照字典序排列。
输入描述:
输入第一行为一个正整数n(1≤n≤1000),下面n行为n个字符串(字符串长度≤100),字符串中只含有大小写字母。
输出描述:
数据输出n行，输出结果为按照字典序排列的字符串。
'''
while True:
    try:
        row = int(input())
        word = []
        for i in range(row):
            word.append(input())
        word = sorted(word)
        for w in word:
            print(w)
    except:
        break   

