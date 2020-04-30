'''
题目描述 
请实现一个函数，将一个字符串中的空格替换成“%20”。 
例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

分析 
将长度为1的空格替换为长度为3的“%20”，字符串的长度变长。 
如果允许我们开辟一个新的数组来存放替换空格后的字符串， 
那么这道题目就非常简单。设置两个指针分别指向新旧字符串首元素， 
遍历原字符串，如果碰到空格就在新字符串上填入“%20”， 
否则就复制元字符串上的内容。但是如果面试官要求 
在原先的字符串上操作，并且保证原字符串有足够长的空间来存放替换后的字符串，

那么我们就得另想方法。 
首先遍历原字符串，找出字符串的长度以及其中的空格数量， 
根据原字符串的长度和空格的数量我们可以求出最后新字符串的长度。
设置两个指针point1和point2分别指向原字符串和新字符串的末尾位置。 
（这里为什么取末尾开始遍历，而不是取起始开始遍历，是为了利用point1==point2这个判断条件） 
如果point1指向内容不为空格，那么将内容赋值给point2指向的位置， 
如果point1指向为空格，那么从point2开始赋值“02%” 
直到point1==point2时表明字符串中的所有空格都已经替换完毕。
'''
class Solution:
    def replaceSpace(self, old_string):
        blank_number = 0 # 空格数量
        old_string_len = len(old_string)

        #遍历原字符串，找出字符串的空格数量
        for i in range(old_string_len):
            if old_string[i] == " ":
                blank_number += 1

        # 计算新字符串的长度
        new_string_len = blank_number * 2 + old_string_len

        # 声明字符串列表
        new_string_list = [" "] * new_string_len

        # 设置两个指针，分别指向那个原理字符串和新字符串的末尾位置
        point1 = old_string_len - 1
        point2 = new_string_len -1
        print(point1)
        # 遍历替换
        while point1 != point2:
            if old_string[point1] != ' ':
                new_string_list[point2] = old_string[point1]
                point1 -= 1
                point2 -= 1
            else:
                new_string_list[point2] = "0"
                new_string_list[point2-1] = "2"
                new_string_list[point2 -2] = "%"
                point1 -= 1
                point2 -= 3

        # 指针相同时，补上原字符
        if point1 > 0:
            for i in range(point1, -1, -1):
                print(i)
                new_string_list[i] = old_string[i]
                print(new_string_list[i])

        # 把字符串数组合成字符串
        new_string = ''
        for i in range(new_string_len):
            new_string += str(new_string_list[i])

        return new_string

    def replaceSpace2(self, s):
        i = 0
        n = len(s)
        ss = [] #装转化字符串
        for i in range(n):
            if s[i].isspace():
                ss.append('%20')
            else:
                ss.append(s[i])
            i += 1
        ss = "".join(ss)
        return ss

if __name__ == "__main__":
    S = Solution()
    print(S.replaceSpace("We Are Happy"))
    print(S.replaceSpace2("We Are Happy"))