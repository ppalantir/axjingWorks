'''
@Author: your name
@Date: 2020-01-14 14:43:34
@LastEditTime : 2020-01-14 14:55:37
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\07_取近似值.py
'''

'''
题目描述
写出一个程序，接受一个正浮点数值，输出该数值的近似整数值。如果小数点后数值大于等于5,向上取整；小于5，则向下取整。

输入描述:
输入一个正浮点数值

输出描述:
输出该数值的近似整数值
'''
while True:
    try:
        f_number = float(input())
        if f_number>=0:
            if f_number - int(f_number)>=0.5:
                f_number = int(f_number)+1
                
            else:
                f_number = int(f_number)
        else:
            if int(f_number) - f_number >= 0.5:
                f_number = int(f_number) - 1
            else:
                f_number = int(f_number)
        print(f_number)
    except:
        break