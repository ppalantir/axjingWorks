import sys


def sulotion(n1, h):
    n = n1[0]
    k = n1[1]

    if h == [] or len(h) == 0:
        return 0
    a1 = 0
    l_ = []
    for i in range(n-k+1):
        a2 = 0
        for j in range(k):
            a2 = a2 + h[i+j]
        l_.append(a2)
    sort_l = sorted(l_)
    print(sort_l)
    return sort_l[0]

if __name__ =="__main__":
    n1 = [7, 3]
    h = [1, 2, 6, 1, 1, 1, 1]
    s = sulotion(n1, h)
    print(s)