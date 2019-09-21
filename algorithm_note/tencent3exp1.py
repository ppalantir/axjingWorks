def Solve(li1, li2):
    max_1 = max(li1)
    li1.remove(max_1)
    li = []
    for i in range(len(li1)):
        for j in range(len(li2)):
            li.append(li1[i] *li2[j])
    return max(li)






while True:
    try:
        n, m = map(int, input().strip().split())
        #print(n, m)
        li_1 = input().strip().split()
        li_2 = input().strip().split()

        li1 = []
        li2 = []
        for i in range(n):
            li1.append(int(li_1[i]))
        for j in range(m):
            li2.append(int(li_2[j]))
        print(Solve(li1, li2))

    except EOFError:
        break
    except ValueError:
        continue
