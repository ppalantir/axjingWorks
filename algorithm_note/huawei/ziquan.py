'''
@Author: your name
@Date: 2020-02-20 19:35:27
@LastEditTime: 2020-02-20 19:49:17
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\ziquan.py
'''

zif = input()
i = 0
j=0
res_l=""
while i < len(zif-1):
    if zif[i] == zif[i+1]:
        res = str(j+1)+zif[i]
    else:
        res = str(1)+zif[i]
    res_l = " ".join(res) 
    i = i+1
    