'''
@Author: your name
@Date: 2020-02-20 17:08:27
@LastEditTime: 2020-02-20 17:08:40
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\坐标移动.py
'''

while True:
    try:
        a=input().strip().split(";")
        point=[0,0]
         
        for i in a:
            if len(i)<2 or len(i)>3 or  not i[1:].isdigit():
                continue
            if i[0]=="A":
                point[0]-=int(i[1:])
            if i[0]=='D':
                point[0]+=int(i[1:])
            if i[0]=='W':
                point[1]+=int(i[1:])
            if i[0]=='S':
                point[1]-=int(i[1:])
        print(str(point[0])+","+str(point[1]))
    except:
            break   