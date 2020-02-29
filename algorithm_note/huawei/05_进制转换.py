'''
@Author: your name
@Date: 2020-01-13 16:29:41
@LastEditTime : 2020-01-13 16:41:26
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\05_进制转换.py
'''
'''
写出一个程序，接受一个十六进制的数，输出该数值的十进制表示。（多组同时输入 ）
'''
while True:
    try:
        number = input()
        n = len(number)
        dic = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'A':10,'B':11,'C':12,'D':13,'E':14,'F':15}
        final = 0
        for i in range(2,n):
            final += dic[number[i]]*(16**(n-i-1))
        print(final)
    except:
        break



while True:
    try:
        print(int(input(),16))
 
    except:
        break
