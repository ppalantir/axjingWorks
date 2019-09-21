
def solve(input1, input2):
    l = []
    for i in input1:
        ln = len(i)
        l.append(ln)
    if input2 in l:
        return input2
    else:
        return max(l)
            



if __name__ == "__main__":
    input1 = input("字符串:").split()
    input2 = input("查找长度：")
    input2 = int(input2)
    n = solve(input1, input2)
    print(n)