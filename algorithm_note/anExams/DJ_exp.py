"""
输入
输入第一行包含一个正整数T，表示数据组数。(1<=T<=10)

接下来有T组输入，每组输入第一行是两个正整数n,m,表示单位迷宫是n行m列（1<=n,m<=300）。

之后n行每行是一个长度为m的字符串。如题所述，描述了一个单位迷宫。

输出
对于每组输入输出一行，如果这个单位迷宫是合法的，就输出“Yes”，否则就输出“No”。


样例输入
2
2 2
S#
#.
3 3
...
###
#S#
样例输出
No
Yes
"""

import sys





input_list = []   
while True:

    print("请输入您的单位迷宫：")
    n = sys.stdin.readline().strip("\n")
    if n == '':
        break
    n = list(map(int, n.split()))
    print(n)
    print("-----")
    input_list.append(n)
print(input_list)


def is_input_legal():
    if input_list[0][0] < 10 and len(input_list[0]) == 1:
        print("第一行合法")
    else:
        print("No")

    if len(input_list[1]) == 2:
        print("第二行合法")
    else:
        print("No")
    print(input_list[2:])
    if len(input_list[2:]) == input_list[1][0] and len(input_list[2:][:]) == input_list[1][1]:
        print("Yes")
    else:
        print("No")
    
    # #while True:
    # for line in sys.stdin:
    #     list_input = line.split()
            
    # return list_input
        # input_1 = input("请输入数据组数T: ")
        # print(input_1)
        # if int(input_1) > 10:
        #     print("No")
        # input_2 = input("请输入")
    
    


if __name__ == "__main__":
    is_input_legal()