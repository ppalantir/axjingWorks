"""
题目描述：
统计string中每个单词出现的个数（不区分大小写）
输入描述：
输入一个字符串，例如："a b A v"
输出描述：
v:1
:3
b:1
a:2
"""
def count_num(s):
    s = s.upper()
    s = ' '.join(s)
    print(s)
    s_list = s.split(' ')
    print(s_list)
    dict_num = {}
    for i in range(26):
        dict_num[chr(i + 65)] = 0

    
    for i in s_list:
        if ord("A")<=ord(i)<=ord("Z"):
            print("%s:%d"%(i, s_list.count(i)))
            dict_num[i] += 1
        else:
            continue
    
    nums = [dict_num[chr(i+65)] for i in range(26)]
    return nums

if __name__ == "__main__":
    s = "aaBBbAcccddl"
    print(count_num(s))
        