# while True:
#     input1 = input("请用户输入n,m:").split(" ")

#     num2 = []
#     for i in range(input1[0]):
#         input2 = input("请输入球门范围a, b：").split(" ")
#         if input2[0]>=1 and input2[1]<=1000:
#             num2.append((input2[0], input2[1]))
#         else:
#             print("输入不在要求范围，请重新输入！！！")
#     num1 = []
#     for i in range(input1[1]):
#         input3 = input("请输入球所在坐标x：")
#         if 1<=x<=1000:
#             num1.append(input3)
#         else:
#             print("输入不在要求范围，请重新输入！！！")


def football(l_input, a):
    c = 0
    li = l_input.copy()
    while c < a-4:
        li.append(li[0+c] + li[1+c], li[3+c])
        c += 1
    return li[-1]


while True:
    try:
        inputP = input("请输入数据：").strip().split()
        l_input = []
        for i in range(len(inputP) - 1):
            l_input.append(int(inputP[i]))

        a = int(inputP[-1])
        o = football(l_input, a)
        print("输出结果：", o)
    except:
        break