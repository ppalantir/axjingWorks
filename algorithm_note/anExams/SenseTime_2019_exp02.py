def calc_ming(n, k):
    allpeople = []
    for i in range(n):
        allpeople.append(i+1)
    peoplenum = n
    while peoplenum > 2:
        removelist = []
        for i in range(len(allpeople)):
            lunshu = int((i + 1) / k )
            if ((i + 1) - lunshu * k) % k == 1:
                removelist.append(allpeople[i])
        for i in range(len(removelist)):
            allpeople.remove(removelist[i])

        peoplenum = len(allpeople)

    return allpeople

while True:
    try:
        c = int(input())
        nlist = []
        klist = []
        for i in range(c):
            nk = input().split()
            n = int(nk[0])
            k = int(nk[1])
            nlist.append(n)
            klist.append(k)
        for i in range(c):
            result = calc_ming(nlist[i], klist[i])
            print(result)

    except:
        break