'''
@Author: your name
@Date: 2020-02-23 14:06:15
@LastEditTime: 2020-02-23 14:09:00
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\简单错误记录.py
'''

'''
题目描述
开发一个简单错误记录功能小模块，能够记录出错的代码所在的文件名称和行号。

处理：
1、 记录最多8条错误记录，循环记录（或者说最后只输出最后出现的八条错误记录），对相同的错误记录（净文件名称和行号完全匹配）只记录一条，错误计数增加；
2、 超过16个字符的文件名称，只记录文件的最后有效16个字符；
3、 输入的文件可能带路径，记录文件名称不能带路径。
'''

error = dict()
filelist = []
while True:
    try:
        record = ' '.join(''.join(input().split('\\')[-1]).split())
        filename = record.split()
        if len(filename[0]) >= 16:
            filename[0] = filename[0][-16:]
        record = ' '.join(filename)
        if record not in error.keys():
            error[record] = 1
            filelist.append(record)
        else:
            error[record] += 1        
    except:
        break
key = filelist[-8:]
for each in key:
    print(' '.join(each.split()),error[each])
