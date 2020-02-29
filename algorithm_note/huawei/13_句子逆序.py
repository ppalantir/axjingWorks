'''
@Author: your name
@Date: 2020-01-14 16:13:34
@LastEditTime : 2020-01-14 16:25:30
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\13_句子逆序.py
'''

'''
题目描述
将一个英文语句以单词为单位逆序排放。例如“I am a boy”，逆序排放后为“boy a am I”
所有单词之间用一个空格隔开，语句中除了英文字母外，不再包含其他字符

接口说明
/**
 * 反转句子
 * 
 * @param sentence 原句子
 * @return 反转后的句子
 */
public String reverse(String sentence);

输入描述:
将一个英文语句以单词为单位逆序排放。

输出描述:
得到逆序的句子
'''

while True:
    try:
        word_char = input().split(" ")
        res = ""
        for i in word_char:
            res = i + ' ' + res
        print(res)
    except:
        break