'''
@Author: your name
@Date: 2020-03-11 10:14:31
@LastEditTime: 2020-03-11 10:24:48
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\99.py
'''


def solution(input_1,input_2):
    input_num = input_1+input_2
    for i in range(len(input_num)):
        if input_num[i] in input_num[i+1:]:
            input_num.pop(input_num[i])
    return input_num

while True:
    input_1 = input()
    input_2 = input()
    input_num = solution(input_1, input_2)
    print(input_num)


    