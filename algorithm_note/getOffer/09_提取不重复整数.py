'''
@Author: your name
@Date: 2020-01-14 15:21:19
@LastEditTime : 2020-01-14 15:30:38
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\09_提取不重复整数.py
'''

'''题目描述
输入一个int型整数，按照从右向左的阅读顺序，返回一个不含重复数字的新的整数。

输入描述:
输入一个int型整数

输出描述:
按照从右向左的阅读顺序，返回一个不含重复数字的新的整数
'''
result = ""
for i in input()[::-1]:
    if i not in result:
        result += i
print(result)