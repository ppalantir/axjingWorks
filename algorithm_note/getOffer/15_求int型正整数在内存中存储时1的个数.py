'''
@Author: your name
@Date: 2020-01-14 16:34:57
@LastEditTime : 2020-01-14 16:39:27
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\15_求int型正整数在内存中存储时1的个数.py
'''

'''
题目描述
输入一个int型的正整数，计算出该int型数据在内存中存储时1的个数。

输入描述:
 输入一个整数（int类型）

输出描述:
 这个数转换成2进制后，输出1的个数
 '''

while True:
    try:
        int_num = int(input())
        int_to_bin = bin(int_num)
        count_ = int_to_bin.count("1")
        print(count_)
    except:
        break
