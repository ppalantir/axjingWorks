"""
输入
共一行，一个字符串s，仅由英文小写字母组成，1≤|s|≤10000。

输出
一个正整数，表示最大出现次数。


样例输入
aba
样例输出
2

提示
aba的所有子串为a、b、a、ab、ba、aba，其中a的出现次数最多，出现了2次。
规则"""

def s_max_num():
    inputs = input("请输入小写字母串：")
    dict_num = {}
    l = []
    for i in inputs:
        c = inputs.count(i)
        dict_num[i] = inputs.count(i)
        
        l.append(c)
    print()
    max_s = 0
    for i in range(len(inputs)):
        max_s = l[i]
        for j in range(i, len(inputs)):
            if max_s < l[j]:
                max_s = l[j]
        
                
    return max_s, max(l)



if __name__ == "__main__":
    print(s_max_num())

    #
    # s_max_num1()