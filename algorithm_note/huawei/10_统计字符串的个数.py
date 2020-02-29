'''
@Author: your name
@Date: 2020-01-14 15:32:31
@LastEditTime : 2020-01-14 15:44:35
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\10_统计字符串的个数.py
'''

'''
题目描述
编写一个函数，计算字符串中含有的不同字符的个数。字符在ACSII码范围内(0~127)，换行表示结束符，不算在字符里。不在范围内的不作统计。

输入描述:
输入N个字符，字符在ACSII码范围内。

输出描述:
输出范围在(0~127)字符的个数。
'''
while True:
    try:
        res = 0
        is_chhecked_char = ""
        for i in input():
            if 0<=ord(i)<=127:
                if i not in is_chhecked_char:
                    is_chhecked_char += i
                    res += 1
        print(res)
    except:
        break
