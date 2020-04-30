'''
@Author: your name
@Date: 2020-01-14 14:56:15
@LastEditTime : 2020-01-14 15:13:57
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\08_合并记录.py
'''

'''
题目描述
数据表记录包含表索引和数值（int范围的整数），请对表索引相同的记录进行合并，即将相同索引的数值进行求和运算，输出按照key值升序进行输出。

输入描述:
先输入键值对的个数
然后输入成对的index和value值，以空格隔开

输出描述:
输出合并后的键值对（多行）
'''
import collections
while True:
    try:
        a, dd = int(input()), collections.defaultdict(int)
        for i in range(a):
            key_, val = map(int, input().split(" "))
            dd[key_] += val
        for i in sorted(dd.keys()):
            print(str(i) + " " + str(dd[i]))
    except:
        break